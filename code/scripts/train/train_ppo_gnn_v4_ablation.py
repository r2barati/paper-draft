#!/usr/bin/env python3
"""
train_ppo_gnn_v4_ablation.py — Ablation study for GNN improvements.

Usage:
  python3 scripts/train/train_ppo_gnn_v4_ablation.py --run A --steps 200000  # V3 baseline
  python3 scripts/train/train_ppo_gnn_v4_ablation.py --run B --steps 200000  # + pipeline feats
  python3 scripts/train/train_ppo_gnn_v4_ablation.py --run C --steps 200000  # + 4 parallel envs
  python3 scripts/train/train_ppo_gnn_v4_ablation.py --run D --steps 200000  # + cosine LR
  python3 scripts/train/train_ppo_gnn_v4_ablation.py --run E --steps 200000  # all combined
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pickle
import argparse
import json
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RescaleAction

from src.envs.core.environment import CoreEnv
from src.envs.wrappers.feature_wrappers import DomainFeatureWrapper, DomainRandomizationWrapper
from src.models.gnn_extractor_v3 import GNNFeaturesExtractorV3

# ---------------------------------------------------------------------------
# Ablation Configurations
# ---------------------------------------------------------------------------

ABLATIONS = {
    'A': {  # Control: V3 baseline
        'desc': 'V3 baseline (control)',
        'pipeline_feats': False,
        'n_envs': 1,
        'cosine_lr': False,
    },
    'B': {  # + pipeline features
        'desc': '+ in-transit & backlog features',
        'pipeline_feats': True,
        'n_envs': 1,
        'cosine_lr': False,
    },
    'C': {  # + parallel envs
        'desc': '+ 4 parallel envs',
        'pipeline_feats': False,
        'n_envs': 4,
        'cosine_lr': False,
    },
    'D': {  # + cosine LR
        'desc': '+ cosine LR decay',
        'pipeline_feats': False,
        'n_envs': 1,
        'cosine_lr': True,
    },
    'E': {  # All combined
        'desc': 'ALL combined (B+C+D)',
        'pipeline_feats': True,
        'n_envs': 4,
        'cosine_lr': True,
    },
}

BASE_PPO_CONFIG = dict(
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=512,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.005,
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=0.03,
)


def cosine_lr_schedule(initial_lr: float = 1e-4, min_lr: float = 1e-5):
    """Returns a callable that computes cosine-decayed LR."""
    def schedule(progress_remaining: float) -> float:
        # progress_remaining goes from 1.0 → 0.0
        return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * (1 - progress_remaining)))
    return schedule


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def make_train_env(pipeline_feats: bool, seed: int = 0):
    def _init():
        env = CoreEnv(
            scenario='base',
            demand_config={'type': 'stationary', 'base_mu': 20, 'use_goodwill': False},
            num_periods=30, backlog=True,
        )
        env = DomainFeatureWrapper(env, is_blind=False, enhanced=True,
                                    grouped=True, pipeline_feats=pipeline_feats)
        env = DomainRandomizationWrapper(env)
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        env = Monitor(env)
        return env
    return _init


def make_eval_env(pipeline_feats: bool, seed: int = 99):
    def _init():
        env = CoreEnv(
            scenario='base',
            demand_config={'type': 'stationary', 'base_mu': 20, 'use_goodwill': False},
            num_periods=30, backlog=True,
        )
        env = DomainFeatureWrapper(env, is_blind=False, enhanced=True,
                                    grouped=True, pipeline_feats=pipeline_feats)
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        env = Monitor(env)
        return env
    return _init


# ---------------------------------------------------------------------------
# Callback: track eval results for easy comparison
# ---------------------------------------------------------------------------

class AblationEvalTracker(BaseCallback):
    """Records eval rewards to a JSON file for cross-run comparison."""
    def __init__(self, run_name: str, log_dir: str, verbose=0):
        super().__init__(verbose)
        self.run_name = run_name
        self.log_file = os.path.join(log_dir, f'ablation_{run_name}_evals.json')
        self.eval_results = []

    def _on_step(self):
        # Check if EvalCallback just logged
        if self.num_timesteps % 5000 == 0:
            infos = self.locals.get('infos', [{}])
            ep_rewards = [info['episode']['r'] for info in infos if 'episode' in info]
            if ep_rewards:
                entry = {'steps': self.num_timesteps, 'reward': float(np.mean(ep_rewards))}
                self.eval_results.append(entry)
        return True

    def _on_training_end(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.eval_results, f, indent=2)


class PrintProgressCallback(BaseCallback):
    def __init__(self, run_name, log_freq=10_000, verbose=0):
        super().__init__(verbose)
        self.run_name = run_name
        self.log_freq = log_freq

    def _on_step(self):
        if self.num_timesteps % self.log_freq == 0:
            infos = self.locals.get('infos', [{}])
            ep_rewards = [info['episode']['r'] for info in infos if 'episode' in info]
            if ep_rewards:
                print(f"  [{self.run_name}] t={self.num_timesteps:7d}  rollout={np.mean(ep_rewards):+.1f}")
        return True


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def train_ablation(run_name: str, total_timesteps: int = 200_000):
    cfg = ABLATIONS[run_name]
    print("=" * 60)
    print(f"  ABLATION RUN {run_name}: {cfg['desc']}")
    print(f"  Steps: {total_timesteps:,} | Envs: {cfg['n_envs']} | Pipeline: {cfg['pipeline_feats']} | CosineLR: {cfg['cosine_lr']}")
    print("=" * 60)

    log_dir = f'./ablation_v4_logs/run_{run_name}/'
    model_path = f'data/models/ablation_v4_{run_name}'
    stats_path = f'data/models/ablation_v4_{run_name}_vecnorm.pkl'
    os.makedirs(log_dir, exist_ok=True)

    # Build envs
    env_fns = [make_train_env(cfg['pipeline_feats'], seed=i) for i in range(cfg['n_envs'])]
    if cfg['n_envs'] > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, gamma=BASE_PPO_CONFIG['gamma'])

    eval_env = DummyVecEnv([make_eval_env(cfg['pipeline_feats'], seed=99)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                             clip_obs=10.0, gamma=BASE_PPO_CONFIG['gamma'], training=False)

    # Determine n_node_feats for the GNN extractor
    n_node_feats = DomainFeatureWrapper.V4_NODE_FEATS if cfg['pipeline_feats'] else DomainFeatureWrapper.V2_NODE_FEATS

    policy_kwargs = dict(
        features_extractor_class=GNNFeaturesExtractorV3,
        features_extractor_kwargs=dict(
            features_dim=256, scenario='base', hidden_dim=64, n_layers=3,
            n_node_feats=n_node_feats,
        ),
        net_arch=[256, 128],
    )

    # PPO config
    ppo_kwargs = BASE_PPO_CONFIG.copy()
    if cfg['cosine_lr']:
        ppo_kwargs['learning_rate'] = cosine_lr_schedule(1e-4, 1e-5)

    model = PPO(
        policy='MlpPolicy', env=vec_env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir, verbose=1,
        **ppo_kwargs,
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Obs dim: {vec_env.observation_space.shape}  |  Params: {total_params:,}")

    eval_cb = EvalCallback(
        eval_env, best_model_save_path=log_dir,
        log_path=log_dir, eval_freq=5_000,  # More frequent for ablation
        n_eval_episodes=5, deterministic=True, verbose=1,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_cb, PrintProgressCallback(run_name), AblationEvalTracker(run_name, log_dir)]
    )

    model.save(model_path)
    with open(stats_path, 'wb') as f:
        pickle.dump(vec_env, f)

    # Print final eval summary
    best_eval = -np.inf
    try:
        import json as _j
        eval_log = np.load(os.path.join(log_dir, 'evaluations.npz'))
        best_idx = np.argmax(eval_log['results'].mean(axis=1))
        best_eval = eval_log['results'][best_idx].mean()
        best_step = eval_log['timesteps'][best_idx]
        print(f"\n  ✅ Run {run_name} DONE | Best eval: {best_eval:+.1f} at step {best_step}")
    except Exception:
        print(f"\n  ✅ Run {run_name} DONE")

    vec_env.close()
    eval_env.close()
    return best_eval


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, required=True, choices=list(ABLATIONS.keys()))
    parser.add_argument('--steps', type=int, default=200_000)
    args = parser.parse_args()
    train_ablation(args.run, args.steps)
