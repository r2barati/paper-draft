#!/usr/bin/env python3
"""
train_gnn_il_bc.py — Behavioral Cloning for GNN-IL.

Trains GNN V3 architecture via supervised MSE loss on Oracle demonstrations.
This gives a warm-started policy that already outputs near-optimal actions.

Usage:
  python3 scripts/train/train_gnn_il_bc.py --epochs 100
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RescaleAction

from src.envs.core.environment import CoreEnv
from src.envs.wrappers.feature_wrappers import DomainFeatureWrapper, DomainRandomizationWrapper
from src.models.gnn_extractor_v3 import GNNFeaturesExtractorV3


def make_env(seed=0):
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


def train_bc(epochs=100, batch_size=256, lr=1e-3):
    # --- Load demonstrations ---
    demos = np.load('data/demos/oracle_demos.npz')
    obs_all = demos['obs']     # (N, obs_dim)
    act_all = demos['actions']  # (N, act_dim)
    print(f"Loaded {obs_all.shape[0]} demo pairs, obs={obs_all.shape[1]}, act={act_all.shape[1]}")

    # --- Create a VecNormalize to compute obs normalization stats ---
    vec_env = DummyVecEnv([make_env(0)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                            clip_obs=10.0, gamma=0.995)

    # Warm up normalization with demo observations
    print("Warming up VecNormalize with demo observations...", flush=True)
    for i in range(min(1000, obs_all.shape[0])):
        vec_env.normalize_obs(obs_all[i:i+1])

    # Normalize all observations
    obs_normed = np.array([vec_env.normalize_obs(obs_all[i:i+1]).squeeze(0)
                           for i in range(obs_all.shape[0])], dtype=np.float32)

    # --- Create PPO model (for architecture, not training) ---
    policy_kwargs = dict(
        features_extractor_class=GNNFeaturesExtractorV3,
        features_extractor_kwargs=dict(
            features_dim=256, scenario='base', hidden_dim=64, n_layers=3,
        ),
        net_arch=[256, 128],
    )

    ppo_model = PPO(
        policy='MlpPolicy', env=vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4, verbose=0,
    )

    total_params = sum(p.numel() for p in ppo_model.policy.parameters())
    print(f"GNN-IL model: {total_params:,} params")

    # --- Extract the actor (action_net) for BC training ---
    policy = ppo_model.policy
    device = policy.device

    obs_tensor = torch.tensor(obs_normed, dtype=torch.float32, device=device)
    act_tensor = torch.tensor(act_all, dtype=torch.float32, device=device)

    dataset = TensorDataset(obs_tensor, act_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # We train the entire policy (features extractor + actor MLP) via MSE
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # --- Behavioral Cloning Training Loop ---
    print(f"\nTraining BC for {epochs} epochs, {len(loader)} batches/epoch", flush=True)
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for obs_batch, act_batch in loader:
            # Get the policy's action prediction (mean of the Gaussian)
            features = policy.extract_features(obs_batch, policy.features_extractor)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            action_mean = policy.action_net(latent_pi)

            loss = criterion(action_mean, act_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: MSE loss = {avg_loss:.6f}", flush=True)

    # --- Save BC model ---
    ppo_model.save('data/models/ppo_gnn_il_bc')
    with open('data/models/vec_normalize_gnn_il.pkl', 'wb') as f:
        pickle.dump(vec_env, f)

    print(f"\n  Saved: data/models/ppo_gnn_il_bc.zip, data/models/vec_normalize_gnn_il.pkl")

    # --- Quick BC-only evaluation ---
    print("\nEvaluating BC-only policy...", flush=True)
    from scripts.train.collect_oracle_demos import SCENARIOS
    eval_seeds = [42, 123, 456]
    profits = []
    for net, dem, gw, bl in SCENARIOS[:4]:  # Just stationary for quick check
        for seed in eval_seeds:
            env = CoreEnv(scenario=net, backlog=bl, num_periods=30,
                          demand_config={'type': dem, 'use_goodwill': gw, 'base_mu': 20})
            env = DomainFeatureWrapper(env, is_blind=False, enhanced=True, grouped=True)
            env = RescaleAction(env, min_action=-1.0, max_action=1.0)
            obs, _ = env.reset(seed=seed)
            ep_r = 0
            for t in range(30):
                obs_n = vec_env.normalize_obs(obs.reshape(1, -1)).squeeze(0)
                action, _ = ppo_model.predict(obs_n, deterministic=True)
                obs, reward, done, trunc, info = env.step(action)
                ep_r += info.get('raw_reward', reward)
                if done or trunc: break
            profits.append(ep_r)
    print(f"  BC-only eval (stationary, 4 configs × 3 seeds): {np.mean(profits):+.1f} +/- {np.std(profits):.1f}")

    vec_env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train_bc(epochs=args.epochs, lr=args.lr)
