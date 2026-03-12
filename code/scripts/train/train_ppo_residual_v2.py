import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
"""
train_ppo_residual_v2.py — Per-Node Shared MLP Residual PPO Training

Trains a PPO agent using the topology-invariant Shared MLP architecture.
The agent learns per-link residual corrections to the Newsvendor heuristic,
with domain randomization for demand scenario generalization.

Pipeline:
  env  → PerLinkFeatureWrapper (14 per-link features)
       → DomainRandomizationWrapper (randomize demand each episode)
       → ResidualActionWrapper (heuristic base + RL residual)
       → DummyVecEnv → VecNormalize → PPO(SharedMLPExtractor)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pickle
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.envs.core.environment import CoreEnv
from src.envs.wrappers.per_link_wrapper import PerLinkFeatureWrapper
from src.envs.wrappers.feature_wrappers import DomainRandomizationWrapper
from src.envs.wrappers.action_wrappers import ProportionalResidualWrapper
from src.agents.heuristic_agent import HeuristicAgent
from src.models.shared_mlp_extractor import SharedMLPExtractor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_ENV_CONFIG = dict(
    scenario='base',
    demand_config={
        'type': 'stationary',
        'base_mu': 20,
        'use_goodwill': False,
    },
    num_periods=30,
    backlog=True,
)

PPO_CONFIG = dict(
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class EpisodeRewardCallback(BaseCallback):
    """Logs mean episode reward to stdout every `log_freq` timesteps."""
    def __init__(self, log_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            infos = self.locals.get('infos', [{}])
            ep_rewards = [info['episode']['r'] for info in infos if 'episode' in info]
            if ep_rewards:
                print(f"  [t={self.num_timesteps:7d}]  mean_ep_reward={np.mean(ep_rewards):.2f}")
        return True

# ---------------------------------------------------------------------------
# Main Training
# ---------------------------------------------------------------------------

def train(total_timesteps: int = 500_000, max_residual: float = 50.0, network: str = 'base'):
    # Parameterize save paths by network topology
    model_save = f'data/models/ppo_residual_v2_{network}'
    stats_save = f'data/models/vec_normalize_residual_v2_{network}.pkl'
    log_dir    = f'./ppo_residual_v2_{network}_logs/'
    os.makedirs(log_dir, exist_ok=True)

    if network != 'base':
        BASE_ENV_CONFIG['scenario'] = network

    # Train Env — Per-Link features + Domain Randomization
    def _make_train():
        env = CoreEnv(**BASE_ENV_CONFIG)
        heuristic = HeuristicAgent(env, is_blind=False)
        env = PerLinkFeatureWrapper(env, heuristic_agent=heuristic, is_blind=False)
        env = DomainRandomizationWrapper(env)
        env = ProportionalResidualWrapper(env, heuristic_agent=heuristic, max_pct=0.5, reward_lambda=0.001)
        env = Monitor(env)
        env.reset(seed=0)
        return env

    # Probe env to get observation shape for extractor config
    probe_env = _make_train()
    obs_shape = probe_env.observation_space.shape
    n_links = obs_shape[0] // PerLinkFeatureWrapper.FEATURES_PER_LINK
    probe_env.close()

    # Configure policy with SharedMLPExtractor
    policy_kwargs = dict(
        features_extractor_class=SharedMLPExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            features_per_link=PerLinkFeatureWrapper.FEATURES_PER_LINK,
            hidden_dim=64,
        ),
        net_arch=[128, 64],  # Smaller head since shared MLP does heavy lifting
    )

    vec_env = DummyVecEnv([_make_train])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=PPO_CONFIG['gamma'],
    )

    # Eval Env — also with Domain Randomization
    def _make_eval():
        cfg = BASE_ENV_CONFIG.copy()
        cfg['demand_config'] = {'type': 'stationary', 'base_mu': 20, 'use_goodwill': False}
        env = CoreEnv(**cfg)
        heuristic = HeuristicAgent(env, is_blind=False)
        env = PerLinkFeatureWrapper(env, heuristic_agent=heuristic, is_blind=False)
        env = DomainRandomizationWrapper(env)
        env = ProportionalResidualWrapper(env, heuristic_agent=heuristic, max_pct=0.5, reward_lambda=0.0)
        env = Monitor(env)
        env.reset(seed=99)
        return env

    eval_vec_env = DummyVecEnv([_make_eval])
    eval_vec_env = VecNormalize(
        eval_vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=PPO_CONFIG['gamma'],
        training=False,
    )

    # Model
    model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        verbose=1,
        **PPO_CONFIG,
    )

    print("=" * 60)
    print(f"Training RESIDUAL PPO V2 (Shared MLP + Domain Randomization)")
    print(f"  Timesteps:       {total_timesteps:,}")
    print(f"  Network:         {BASE_ENV_CONFIG['scenario']}")
    print(f"  Reorder Links:   {n_links}")
    print(f"  Max Residual:    ±{max_residual}")
    print(f"  Obs shape:       {vec_env.observation_space.shape}")
    print(f"  Act shape:       {vec_env.action_space.shape}")
    print(f"  Per-link feats:  {PerLinkFeatureWrapper.FEATURES_PER_LINK}")
    print(f"  Save path:       {model_save}.zip")
    print("=" * 60)

    eval_callback = EvalCallback(
        eval_vec_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    reward_callback = EpisodeRewardCallback(log_freq=10_000)

    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, reward_callback])

    model.save(model_save)
    print(f"\nModel saved → {model_save}.zip")

    with open(stats_save, 'wb') as f:
        pickle.dump(vec_env, f)
    print(f"VecNormalize stats saved → {stats_save}")

    # Save shared MLP weights for cross-topology transfer
    shared_mlp_path = f'data/models/shared_mlp_residual_{network}.pt'
    import torch
    extractor = model.policy.features_extractor
    torch.save(extractor.get_shared_mlp_state(), shared_mlp_path)
    print(f"Shared MLP weights saved → {shared_mlp_path}")

    vec_env.close()
    eval_vec_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=500_000)
    parser.add_argument('--max_residual', type=float, default=50.0)
    parser.add_argument('--network', type=str, default='base')
    args = parser.parse_args()

    train(args.timesteps, args.max_residual, args.network)
