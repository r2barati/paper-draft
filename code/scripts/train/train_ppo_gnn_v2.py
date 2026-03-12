import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
"""
train_ppo_gnn_v2.py — Train GNN V2 (Enhanced MPNN Architecture)

Improvements over V1:
  - 8 per-node features (inv_pos, lt_target, gap, on_hand, h_cost, capacity, is_factory, is_retail)
  - 10 global features (demand_vel, norm_time, sin/cos time, demand_hist[5], goodwill)
  - Edge-conditioned MPNN with directed graph and GAT attention
  - 3 MPNN layers with residual connections
"""

import os
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
from src.models.gnn_extractor_v2 import GNNFeaturesExtractorV2

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
    batch_size=256,
    n_epochs=10,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.005,
    vf_coef=0.5,
    max_grad_norm=0.5,
)

POLICY_KWARGS = dict(
    features_extractor_class=GNNFeaturesExtractorV2,
    features_extractor_kwargs=dict(
        features_dim=256,
        scenario=BASE_ENV_CONFIG['scenario'],
        hidden_dim=64,
        n_layers=3,
    ),
    net_arch=[256, 128],
)

MODEL_SAVE_PATH = 'data/models/ppo_gnn_v2'
STATS_SAVE_PATH = 'data/models/vec_normalize_gnn_v2.pkl'
LOG_DIR         = './ppo_gnn_v2_logs/'

# ---------------------------------------------------------------------------
# Env Factories
# ---------------------------------------------------------------------------

def make_train_env(seed: int = 0):
    def _init():
        env = CoreEnv(**BASE_ENV_CONFIG)
        env = DomainFeatureWrapper(env, is_blind=False, enhanced=True)  # V2 features
        env = DomainRandomizationWrapper(env)
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init

def make_eval_env(demand_type='stationary', use_goodwill=False, seed: int = 99):
    def _init():
        cfg = BASE_ENV_CONFIG.copy()
        cfg['demand_config'] = {
            'type': demand_type,
            'base_mu': 20,
            'use_goodwill': use_goodwill,
        }
        env = CoreEnv(**cfg)
        env = DomainFeatureWrapper(env, is_blind=False, enhanced=True)  # V2 features
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class EpisodeRewardCallback(BaseCallback):
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
# Training
# ---------------------------------------------------------------------------

def train(total_timesteps: int = 1_000_000, network: str = 'base'):
    os.makedirs(LOG_DIR, exist_ok=True)
    if network != 'base':
        BASE_ENV_CONFIG['scenario'] = network
        POLICY_KWARGS['features_extractor_kwargs']['scenario'] = network

    vec_env = DummyVecEnv([make_train_env(seed=0)])
    vec_env = VecNormalize(
        vec_env, norm_obs=True, norm_reward=True,
        clip_obs=10.0, gamma=PPO_CONFIG['gamma'],
    )

    eval_vec_env = DummyVecEnv([make_eval_env('stationary', False, seed=99)])
    eval_vec_env = VecNormalize(
        eval_vec_env, norm_obs=True, norm_reward=False,
        clip_obs=10.0, gamma=PPO_CONFIG['gamma'], training=False,
    )

    model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        policy_kwargs=POLICY_KWARGS,
        tensorboard_log=LOG_DIR,
        verbose=1,
        **PPO_CONFIG,
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print("=" * 60)
    print(f"Training GNN V2 (MPNN) for {total_timesteps:,} timesteps on '{network}'")
    print(f"  Obs dim: {vec_env.observation_space.shape}")
    print(f"  Act dim: {vec_env.action_space.shape}")
    print(f"  Total parameters: {total_params:,}")
    print("=" * 60)

    eval_callback = EvalCallback(
        eval_vec_env, best_model_save_path=LOG_DIR,
        log_path=LOG_DIR, eval_freq=10_000,
        n_eval_episodes=5, deterministic=True, verbose=1,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, EpisodeRewardCallback(10_000)]
    )

    model.save(MODEL_SAVE_PATH)
    with open(STATS_SAVE_PATH, 'wb') as f:
        pickle.dump(vec_env, f)

    vec_env.close()
    eval_vec_env.close()
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--network', type=str, default='base')
    args = parser.parse_args()
    train(total_timesteps=args.timesteps, network=args.network)
