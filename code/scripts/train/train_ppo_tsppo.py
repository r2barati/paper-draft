#!/usr/bin/env python3
"""
train_ppo_tsppo.py — Transformer-Sequential PPO (TSPPO).

Uses TransformerFeaturesExtractor instead of GNN for supply chain RL.
Captures temporal dependencies via self-attention that GNN's spatial message passing cannot.

Usage:
  python3 scripts/train/train_ppo_tsppo.py --steps 200000
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RescaleAction

from src.envs.core.environment import CoreEnv
from src.envs.wrappers.feature_wrappers import DomainFeatureWrapper, DomainRandomizationWrapper
from src.models.transformer_extractor import TransformerFeaturesExtractor


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


def train(total_timesteps=200_000):
    log_dir = './tsppo_logs/'
    os.makedirs(log_dir, exist_ok=True)

    vec_env = DummyVecEnv([make_train_env(0)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, gamma=0.995)

    eval_env = DummyVecEnv([make_eval_env(99)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                             clip_obs=10.0, gamma=0.995, training=False)

    policy_kwargs = dict(
        features_extractor_class=TransformerFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            scenario='base',
            d_model=64,
            n_heads=4,
            n_layers=3,
            dropout=0.1,
        ),
        net_arch=[256, 128],
    )

    model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
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
        tensorboard_log=log_dir,
        verbose=1,
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print("=" * 60)
    print(f"  TSPPO — Transformer-Sequential PPO")
    print(f"  Steps: {total_timesteps:,} | Params: {total_params:,}")
    print(f"  Architecture: Transformer (d=64, heads=4, layers=3)")
    print(f"  Obs dim: {vec_env.observation_space.shape}")
    print("=" * 60)

    eval_cb = EvalCallback(
        eval_env, best_model_save_path=log_dir,
        log_path=log_dir, eval_freq=5_000,
        n_eval_episodes=5, deterministic=True, verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_cb)

    model.save('data/models/ppo_tsppo')

    import pickle
    with open('data/models/vec_normalize_tsppo.pkl', 'wb') as f:
        pickle.dump(vec_env, f)

    # Report
    try:
        eval_log = np.load(os.path.join(log_dir, 'evaluations.npz'))
        best_idx = np.argmax(eval_log['results'].mean(axis=1))
        print(f"\n  ✅ TSPPO DONE | Best: {eval_log['results'][best_idx].mean():+.1f} "
              f"at step {eval_log['timesteps'][best_idx]}")
    except:
        print("\n  ✅ TSPPO DONE")

    vec_env.close()
    eval_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200_000)
    args = parser.parse_args()
    train(args.steps)
