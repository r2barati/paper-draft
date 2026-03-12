#!/usr/bin/env python3
"""
train_ppo_gnn_v3_specialist.py — Train specialist GNN V3 for a specific demand type.

Usage:
  python3 scripts/train/train_ppo_gnn_v3_specialist.py --demand stationary
  python3 scripts/train/train_ppo_gnn_v3_specialist.py --demand trend
  python3 scripts/train/train_ppo_gnn_v3_specialist.py --demand seasonal
  python3 scripts/train/train_ppo_gnn_v3_specialist.py --demand shock

Each specialist is trained on ONE demand type only (no DomainRandomization),
making it a fair comparison against MSSP which also knows the demand type.

Goodwill is randomized via a lightweight wrapper (not a per-step callback).
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
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RescaleAction

from src.envs.core.environment import CoreEnv
from src.envs.wrappers.feature_wrappers import DomainFeatureWrapper
from src.models.gnn_extractor_v3 import GNNFeaturesExtractorV3

# Same stabilized hyperparams as V3 generalist
PPO_CONFIG = dict(
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


class GoodwillRandomWrapper(gym.Wrapper):
    """
    Lightweight wrapper that randomly toggles goodwill on each reset.
    No per-step overhead — only runs on episode boundaries.
    """
    def __init__(self, env):
        super().__init__(env)
        self._rng = np.random.default_rng(42)

    def reset(self, **kwargs):
        # Toggle goodwill randomly
        base = self.env
        while hasattr(base, 'env'):
            base = base.env
        if hasattr(base, 'demand_engine'):
            base.demand_engine.use_goodwill = bool(self._rng.choice([False, True]))
            base.demand_engine.sentiment = 1.0
        return self.env.reset(**kwargs)


class EpisodeRewardCallback(BaseCallback):
    def __init__(self, log_freq=10_000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self):
        if self.num_timesteps % self.log_freq == 0:
            infos = self.locals.get('infos', [{}])
            ep_rewards = [info['episode']['r'] for info in infos if 'episode' in info]
            if ep_rewards:
                print(f"  [t={self.num_timesteps:7d}]  mean_ep_reward={np.mean(ep_rewards):.2f}")
        return True


def make_train_env(demand_type, seed=0):
    def _init():
        env = CoreEnv(
            scenario='base',
            demand_config={'type': demand_type, 'base_mu': 20, 'use_goodwill': False},
            num_periods=30,
            backlog=True,
        )
        env = DomainFeatureWrapper(env, is_blind=False, enhanced=True, grouped=True)
        env = GoodwillRandomWrapper(env)  # Lightweight goodwill toggle
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        env = Monitor(env)
        return env
    return _init


def make_eval_env(demand_type, seed=99):
    def _init():
        env = CoreEnv(
            scenario='base',
            demand_config={'type': demand_type, 'base_mu': 20, 'use_goodwill': False},
            num_periods=30,
            backlog=True,
        )
        env = DomainFeatureWrapper(env, is_blind=False, enhanced=True, grouped=True)
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        env = Monitor(env)
        return env
    return _init


def train_specialist(demand_type: str, timesteps: int = 1_000_000):
    log_dir = f'./ppo_gnn_v3_{demand_type}_logs/'
    model_path = f'data/models/ppo_gnn_v3_{demand_type}'
    stats_path = f'data/models/vec_normalize_gnn_v3_{demand_type}.pkl'
    os.makedirs(log_dir, exist_ok=True)

    vec_env = DummyVecEnv([make_train_env(demand_type, seed=0)])
    vec_env = VecNormalize(
        vec_env, norm_obs=True, norm_reward=True,
        clip_obs=10.0, gamma=PPO_CONFIG['gamma'],
    )

    eval_env = DummyVecEnv([make_eval_env(demand_type, seed=99)])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False,
        clip_obs=10.0, gamma=PPO_CONFIG['gamma'], training=False,
    )

    policy_kwargs = dict(
        features_extractor_class=GNNFeaturesExtractorV3,
        features_extractor_kwargs=dict(
            features_dim=256, scenario='base', hidden_dim=64, n_layers=3,
        ),
        net_arch=[256, 128],
    )

    model = PPO(
        policy='MlpPolicy', env=vec_env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir, verbose=1,
        **PPO_CONFIG,
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print("=" * 60)
    print(f"Training GNN V3 SPECIALIST [{demand_type.upper()}]")
    print(f"  {timesteps:,} timesteps  |  {total_params:,} params")
    print(f"  Demand: {demand_type} (specialist — no domain randomization)")
    print("=" * 60)

    eval_cb = EvalCallback(
        eval_env, best_model_save_path=log_dir,
        log_path=log_dir, eval_freq=10_000,
        n_eval_episodes=5, deterministic=True, verbose=1,
    )

    model.learn(
        total_timesteps=timesteps,
        callback=[eval_cb, EpisodeRewardCallback(10_000)]
    )

    model.save(model_path)
    with open(stats_path, 'wb') as f:
        pickle.dump(vec_env, f)
    vec_env.close()
    eval_env.close()
    print(f"\n  Saved: {model_path}.zip, {stats_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demand', type=str, required=True,
                        choices=['stationary', 'trend', 'seasonal', 'shock'])
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    args = parser.parse_args()
    train_specialist(args.demand, args.timesteps)
