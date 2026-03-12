#!/usr/bin/env python3
"""
train_gnn_il_finetune.py — PPO fine-tuning of BC-pretrained GNN-IL.

Loads the BC-pretrained policy and fine-tunes with PPO.
Uses lower LR (1e-5) to avoid catastrophic forgetting.

Usage:
  python3 scripts/train/train_gnn_il_finetune.py --steps 200000
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
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RescaleAction

from src.envs.core.environment import CoreEnv
from src.envs.wrappers.feature_wrappers import DomainFeatureWrapper, DomainRandomizationWrapper
from src.models.gnn_extractor_v3 import GNNFeaturesExtractorV3


PPO_CONFIG = dict(
    learning_rate=1e-5,       # Lower LR to avoid forgetting BC knowledge
    n_steps=2048,
    batch_size=512,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.15,          # Tighter clip to stay near BC policy
    ent_coef=0.002,           # Lower entropy — BC already knows a good policy
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=0.02,           # Tighter KL — conservative updates
)


def make_train_env(seed=0):
    def _init():
        env = CoreEnv(
            scenario='base',
            demand_config={'type': 'stationary', 'base_mu': 20, 'use_goodwill': False},
            num_periods=30, backlog=True,
        )
        env = DomainFeatureWrapper(env, is_blind=False, enhanced=True, grouped=True)
        env = DomainRandomizationWrapper(env)
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        env = Monitor(env)
        return env
    return _init


def make_eval_env(seed=99):
    def _init():
        env = CoreEnv(
            scenario='base',
            demand_config={'type': 'stationary', 'base_mu': 20, 'use_goodwill': False},
            num_periods=30, backlog=True,
        )
        env = DomainFeatureWrapper(env, is_blind=False, enhanced=True, grouped=True)
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        env = Monitor(env)
        return env
    return _init


class PrintProgressCallback(BaseCallback):
    def __init__(self, log_freq=10_000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self):
        if self.num_timesteps % self.log_freq == 0:
            infos = self.locals.get('infos', [{}])
            ep_rewards = [info['episode']['r'] for info in infos if 'episode' in info]
            if ep_rewards:
                print(f"  [GNN-IL] t={self.num_timesteps:7d}  rollout={np.mean(ep_rewards):+.1f}", flush=True)
        return True


def finetune(total_timesteps=200_000):
    log_dir = './gnn_il_logs/'
    os.makedirs(log_dir, exist_ok=True)

    # --- Load BC-pretrained model ---
    print("Loading BC-pretrained model...", flush=True)
    bc_model = PPO.load(
        'data/models/ppo_gnn_il_bc',
        custom_objects={'GNNFeaturesExtractorV3': GNNFeaturesExtractorV3},
    )

    # --- Load VecNormalize stats from BC ---
    with open('data/models/vec_normalize_gnn_il.pkl', 'rb') as f:
        bc_vecnorm = pickle.load(f)

    # --- Create fresh training env with same normalization ---
    vec_env = DummyVecEnv([make_train_env(0)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, gamma=PPO_CONFIG['gamma'])

    # Copy BC normalization stats to new env
    vec_env.obs_rms = bc_vecnorm.obs_rms
    vec_env.ret_rms = bc_vecnorm.ret_rms

    eval_env = DummyVecEnv([make_eval_env(99)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                             clip_obs=10.0, gamma=PPO_CONFIG['gamma'], training=False)
    eval_env.obs_rms = bc_vecnorm.obs_rms

    # --- Create new PPO with BC weights ---
    model = PPO(
        policy='MlpPolicy', env=vec_env,
        policy_kwargs=dict(
            features_extractor_class=GNNFeaturesExtractorV3,
            features_extractor_kwargs=dict(
                features_dim=256, scenario='base', hidden_dim=64, n_layers=3,
            ),
            net_arch=[256, 128],
        ),
        tensorboard_log=log_dir, verbose=1,
        **PPO_CONFIG,
    )

    # Copy BC weights into the new PPO model
    model.policy.load_state_dict(bc_model.policy.state_dict())

    total_params = sum(p.numel() for p in model.policy.parameters())
    print("=" * 60)
    print(f"  GNN-IL Fine-tuning (BC → PPO)")
    print(f"  {total_timesteps:,} steps  |  {total_params:,} params")
    print(f"  LR: {PPO_CONFIG['learning_rate']}  |  clip: {PPO_CONFIG['clip_range']}  |  target_kl: {PPO_CONFIG['target_kl']}")
    print("=" * 60)

    eval_cb = EvalCallback(
        eval_env, best_model_save_path=log_dir,
        log_path=log_dir, eval_freq=5_000,
        n_eval_episodes=5, deterministic=True, verbose=1,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_cb, PrintProgressCallback()]
    )

    model.save('data/models/ppo_gnn_il')
    with open('data/models/vec_normalize_gnn_il_ft.pkl', 'wb') as f:
        pickle.dump(vec_env, f)

    # Final summary
    try:
        eval_log = np.load(os.path.join(log_dir, 'evaluations.npz'))
        best_idx = np.argmax(eval_log['results'].mean(axis=1))
        best_eval = eval_log['results'][best_idx].mean()
        best_step = eval_log['timesteps'][best_idx]
        print(f"\n  ✅ GNN-IL Fine-tuning DONE | Best eval: {best_eval:+.1f} at step {best_step}")
    except Exception:
        print(f"\n  ✅ GNN-IL Fine-tuning DONE")

    vec_env.close()
    eval_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200_000)
    args = parser.parse_args()
    finetune(args.steps)
