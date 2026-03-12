import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
"""
train_rl.py — PPO Training Pipeline for CoreEnv

Pipeline:
  env  →  RescaleAction([-1,1] → [0, high])
       →  DummyVecEnv
       →  VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10.0)
       →  PPO(policy_kwargs=dict(net_arch=[256, 256]))

Saves:
  ppo_baseline.zip       — trained PPO model
  vec_normalize.pkl      — VecNormalize running stats
"""

import os
import pickle

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RescaleAction

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.envs.core.environment import CoreEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV_CONFIG = dict(
    scenario='base',
    demand_config={
        'type': 'stationary',
        'base_mu': 20,
        'use_goodwill': False,
    },
    num_periods=30,
    backlog=True,
)

TOTAL_TIMESTEPS  = 200_000
POLICY_KWARGS    = dict(net_arch=[256, 256])
MODEL_SAVE_PATH  = 'data/models/ppo_baseline'         # SB3 appends .zip automatically
STATS_SAVE_PATH  = 'data/models/vec_normalize_baseline.pkl'
LOG_DIR          = './ppo_logs/'

# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def make_env(seed: int = 0):
    """
    Creates one training environment instance with the full wrapper stack.
    Returns a *raw* Gymnasium env (wrapping happens in make_vec_env below).
    """
    def _init():
        env = CoreEnv(**ENV_CONFIG)

        # --- Wrapper 1: RescaleAction ---
        # Maps the policy's [-1, 1] output to [0, action_space.high].
        # action_space.low is already 0, so this is exact and zero-clipping-safe.
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)

        # Monitor wraps for episode logging (reward, length)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------

class EpisodeRewardCallback(BaseCallback):
    """Logs mean episode reward to stdout every `log_freq` timesteps."""

    def __init__(self, log_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            infos = self.locals.get('infos', [{}])
            ep_rewards = [
                info['episode']['r']
                for info in infos
                if 'episode' in info
            ]
            if ep_rewards:
                print(f"  [t={self.num_timesteps:7d}]  "
                      f"mean_ep_reward={np.mean(ep_rewards):.2f}")
        return True


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train():
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- Wrapper 2: VecNormalize ---
    # norm_obs=True   : normalises obs to ~N(0,1) using running statistics
    # norm_reward=True: normalises reward similarly (disabled at eval time)
    # clip_obs=10.0   : clips normalised obs to [-10, 10] for stability
    vec_env = DummyVecEnv([make_env(seed=0)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=0.99,
    )

    # --- PPO Model ---
    model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,          # small entropy bonus to encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=POLICY_KWARGS,
        tensorboard_log=LOG_DIR,
        verbose=1,
    )

    print("=" * 60)
    print(f"Training PPO for {TOTAL_TIMESTEPS:,} timesteps")
    print(f"  Network:    {ENV_CONFIG['scenario']}")
    print(f"  Demand:     {ENV_CONFIG['demand_config']['type']}")
    print(f"  Obs shape:  {vec_env.observation_space.shape}")
    print(f"  Act shape:  {vec_env.action_space.shape}")
    print("=" * 60)

    callback = EpisodeRewardCallback(log_freq=10_000)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    # --- Save model ---
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved → {MODEL_SAVE_PATH}.zip")

    # --- Save VecNormalize stats ---
    # We save the raw stats object (running mean/var) so RLAgent can reload
    # them at evaluation time without re-running the full VecEnv machinery.
    with open(STATS_SAVE_PATH, 'wb') as f:
        pickle.dump(vec_env, f)
    print(f"VecNormalize stats saved → {STATS_SAVE_PATH}")

    vec_env.close()
    return model


if __name__ == '__main__':
    train()
