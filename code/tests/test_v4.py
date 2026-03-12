import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
test_v4.py — Diagnostics for PPO V4 Agent

Runs a single episode of the trained V4 agent on the base/stationary
environment, printing out specific step-by-step actions and inventory
levels to determine if action saturation has been fixed.
"""

import os
# macOS thread safety locks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RescaleAction

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.core.environment import CoreEnv
from src.envs.wrappers.feature_wrappers import DomainFeatureWrapper

def test_v4_agent():
    # 1. Initialize the raw base + stationary environment
    env_kwargs = dict(
        scenario='base',
        demand_config={
            'type': 'stationary',
            'base_mu': 20,
            'use_goodwill': False,
        },
        num_periods=30,
        backlog=True
    )
    
    def make_env():
        # 2. Wrap it with DomainFeatureWrapper and RescaleAction
        env = CoreEnv(**env_kwargs)
        env = DomainFeatureWrapper(env, is_blind=False)
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        return env

    # 3. Load the DummyVecEnv and vec_normalize_v4.pkl
    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load("data/models/vec_normalize_v4.pkl", vec_env)
    
    # CRITICAL: training=False, norm_reward=False
    vec_env.training = False
    vec_env.norm_reward = False

    # 4. Load the ppo_v4_supply_chain.zip model (device="cpu")
    model = PPO.load("data/models/ppo_v4_supply_chain.zip", env=vec_env, device="cpu")

    # Access the raw un-vectorized environment to read state
    raw_env = vec_env.envs[0].unwrapped

    print("=" * 60)
    print("Starting V4 Diagnostics (1 Episode)")
    print("=" * 60)

    # 5. Run exactly 1 episode (30 periods)
    obs = vec_env.reset()
    total_reward = 0.0

    # In a loop, print the action array, inventory levels (from env.X), and reward
    for step in range(30):
        # Predict action
        action, _states = model.predict(obs, deterministic=True)
        
        # Take step
        obs, reward, done, info = vec_env.step(action)
        
        # Accumulate reward (Note: vec_env returns an array of rewards)
        step_reward = reward[0]
        total_reward += step_reward
        
        print(f"--- Period {step + 1} ---")
        print(f"Action Output:      {np.round(action[0], 2)}")
        print(f"Inventory (env.X):  {raw_env.X}")
        print(f"Period Reward:      {step_reward:.2f}")
        print()

        if done[0]:
            break

    print("=" * 60)
    print(f"Episode Completed. Total Accumulated Reward: {total_reward:.2f}")
    print("=" * 60)

if __name__ == "__main__":
    test_v4_agent()
