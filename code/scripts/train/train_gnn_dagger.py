#!/usr/bin/env python3
"""
train_gnn_dagger.py — DAgger (Dataset Aggregation) for GNN Imitation Learning.

DAgger fixes standard BC's distribution shift by iteratively:
  1. Roll out current policy in the env → collect states the POLICY visits
  2. Query Oracle: "what would you do at THIS state?" → get expert labels
  3. Aggregate new (obs, oracle_action) pairs with existing dataset
  4. Retrain the policy on the full aggregated dataset

This means the model learns to recover from ITS OWN mistakes, not just Oracle's trajectory.

Usage:
  python3 scripts/train/train_gnn_dagger.py --rounds 10 --epochs 50 --rollouts 20
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pickle
import argparse
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RescaleAction

from src.envs.core.environment import CoreEnv
from src.envs.wrappers.feature_wrappers import DomainFeatureWrapper
from src.agents.oracle import ClairvoyantOracle
from src.models.gnn_extractor_v3 import GNNFeaturesExtractorV3

H = 30
SCENARIOS = list(itertools.product(
    ['stationary', 'trend', 'seasonal', 'shock'],
    [False, True],   # goodwill
    [False, True],   # backlog
))


def make_env_for_vecnorm():
    """Create a dummy env for VecNormalize initialization."""
    def _init():
        env = CoreEnv(scenario='base',
                      demand_config={'type': 'stationary', 'base_mu': 20, 'use_goodwill': False},
                      num_periods=H, backlog=True)
        env = DomainFeatureWrapper(env, is_blind=False, enhanced=True, grouped=True)
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        env = Monitor(env)
        return env
    return _init


def collect_oracle_demos(seeds, scenarios=None):
    """Phase 0: Collect initial Oracle demonstrations (same as standard BC)."""
    if scenarios is None:
        scenarios = SCENARIOS
    all_obs, all_actions = [], []

    for dem, gw, bl in scenarios:
        env_kwargs = {'scenario': 'base', 'backlog': bl, 'num_periods': H,
                      'demand_config': {'type': dem, 'use_goodwill': gw, 'base_mu': 20}}
        for seed in seeds:
            oracle = ClairvoyantOracle(env_kwargs, planning_horizon=H)
            oracle_actions, _ = oracle.solve(seed=seed)

            env = CoreEnv(**env_kwargs)
            env = DomainFeatureWrapper(env, is_blind=False, enhanced=True, grouped=True)
            obs, _ = env.reset(seed=seed)

            for t in range(H):
                raw_action = oracle_actions[t]
                act_low = env.unwrapped.action_space.low
                act_high = env.unwrapped.action_space.high
                rescaled = 2.0 * (raw_action - act_low) / (act_high - act_low + 1e-8) - 1.0
                rescaled = np.clip(rescaled, -1.0, 1.0)

                all_obs.append(obs.astype(np.float32))
                all_actions.append(rescaled.astype(np.float32))

                obs, _, done, trunc, _ = env.step(raw_action)
                if done or trunc:
                    break

    return np.stack(all_obs), np.stack(all_actions)


def dagger_rollout(model, vec_norm, seeds, scenarios=None, beta=0.0):
    """
    DAgger rollout: execute the learned policy, but label each state with Oracle's action.

    Args:
        model: current PPO model (used for rollout)
        vec_norm: VecNormalize for observation normalization
        seeds: list of seeds
        scenarios: list of (dem, gw, bl) tuples
        beta: mixing ratio — probability of using Oracle action instead of policy
              (beta=1.0 = pure Oracle, beta=0.0 = pure policy rollout)
    """
    if scenarios is None:
        scenarios = SCENARIOS
    all_obs, all_actions = [], []

    for dem, gw, bl in scenarios:
        env_kwargs = {'scenario': 'base', 'backlog': bl, 'num_periods': H,
                      'demand_config': {'type': dem, 'use_goodwill': gw, 'base_mu': 20}}

        for seed in seeds:
            # Get Oracle's optimal plan for this episode
            oracle = ClairvoyantOracle(env_kwargs, planning_horizon=H)
            oracle_actions, _ = oracle.solve(seed=seed)

            # Create env with feature wrapper
            env = CoreEnv(**env_kwargs)
            env = DomainFeatureWrapper(env, is_blind=False, enhanced=True, grouped=True)
            env_rescaled = RescaleAction(env, min_action=-1.0, max_action=1.0)
            obs, _ = env_rescaled.reset(seed=seed)

            for t in range(H):
                # Record the observation the POLICY sees
                all_obs.append(obs.astype(np.float32))

                # Get Oracle's action (the expert label)
                raw_oracle = oracle_actions[t]
                act_low = env.unwrapped.action_space.low
                act_high = env.unwrapped.action_space.high
                rescaled_oracle = 2.0 * (raw_oracle - act_low) / (act_high - act_low + 1e-8) - 1.0
                rescaled_oracle = np.clip(rescaled_oracle, -1.0, 1.0)
                all_actions.append(rescaled_oracle.astype(np.float32))

                # Decide which action to actually EXECUTE
                if np.random.random() < beta:
                    # Use Oracle's action (safer early on)
                    obs, _, done, trunc, _ = env_rescaled.step(rescaled_oracle)
                else:
                    # Use policy's action (DAgger: visit policy's states)
                    obs_n = vec_norm.normalize_obs(obs.reshape(1, -1)).squeeze(0)
                    policy_action, _ = model.predict(obs_n, deterministic=True)
                    obs, _, done, trunc, _ = env_rescaled.step(policy_action)

                if done or trunc:
                    break

    return np.stack(all_obs), np.stack(all_actions)


def train_policy(policy, obs_all, act_all, vec_norm, epochs=50, batch_size=256, lr=1e-3):
    """Train the policy via supervised MSE on the aggregated dataset."""
    device = policy.device

    # Normalize observations
    obs_normed = np.array([vec_norm.normalize_obs(obs_all[i:i+1]).squeeze(0)
                           for i in range(obs_all.shape[0])], dtype=np.float32)

    obs_t = torch.tensor(obs_normed, dtype=torch.float32, device=device)
    act_t = torch.tensor(act_all, dtype=torch.float32, device=device)

    loader = DataLoader(TensorDataset(obs_t, act_t), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()

    final_loss = 0
    for epoch in range(epochs):
        total_loss, n = 0, 0
        for obs_b, act_b in loader:
            features = policy.extract_features(obs_b, policy.features_extractor)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            pred = policy.action_net(latent_pi)

            loss = criterion(pred, act_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n += 1
        final_loss = total_loss / n
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}: MSE = {final_loss:.6f}", flush=True)

    return final_loss


def evaluate_policy(model, vec_norm, seeds=[42, 123, 456]):
    """Quick eval on stationary, no-goodwill, backlog=True."""
    env_kwargs = {'scenario': 'base', 'backlog': True, 'num_periods': H,
                  'demand_config': {'type': 'stationary', 'use_goodwill': False, 'base_mu': 20}}
    profits = []
    for seed in seeds:
        env = CoreEnv(**env_kwargs)
        env = DomainFeatureWrapper(env, is_blind=False, enhanced=True, grouped=True)
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        obs, _ = env.reset(seed=seed)
        ep_r = 0
        for t in range(H):
            obs_n = vec_norm.normalize_obs(obs.reshape(1, -1)).squeeze(0)
            action, _ = model.predict(obs_n, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            ep_r += info.get('raw_reward', reward)
            if done or trunc:
                break
        profits.append(ep_r)
    return np.mean(profits), np.std(profits)


def main(n_rounds=10, epochs_per_round=50, rollouts_per_round=20, lr=1e-3):
    print("=" * 60)
    print("  GNN-DAgger — Iterative Imitation Learning")
    print(f"  Rounds: {n_rounds} | Epochs/round: {epochs_per_round}")
    print(f"  Rollouts/round: {rollouts_per_round} seeds × 16 scenarios")
    print("=" * 60)

    os.makedirs('data/models', exist_ok=True)

    # --- Initialize VecNormalize ---
    vec_env = DummyVecEnv([make_env_for_vecnorm()])
    vec_norm = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.995)

    # --- Initialize PPO model (for architecture) ---
    policy_kwargs = dict(
        features_extractor_class=GNNFeaturesExtractorV3,
        features_extractor_kwargs=dict(
            features_dim=256, scenario='base', hidden_dim=64, n_layers=3,
        ),
        net_arch=[256, 128],
    )
    ppo_model = PPO('MlpPolicy', vec_env, policy_kwargs=policy_kwargs,
                     learning_rate=1e-4, verbose=0)
    policy = ppo_model.policy
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"  Model: {total_params:,} params\n")

    # --- Phase 0: Initial Oracle Demos (seeds 0-4) ---
    print("Phase 0: Collecting initial Oracle demonstrations...", flush=True)
    initial_seeds = list(range(5))
    obs_agg, act_agg = collect_oracle_demos(initial_seeds)
    print(f"  Initial dataset: {obs_agg.shape[0]} pairs\n")

    # Warm up VecNormalize with collected observations
    for i in range(min(2000, obs_agg.shape[0])):
        vec_norm.normalize_obs(obs_agg[i:i+1])

    # --- DAgger Loop ---
    best_eval = -np.inf
    for rnd in range(n_rounds):
        print(f"{'='*60}")
        print(f"  DAgger Round {rnd+1}/{n_rounds}  |  Dataset: {obs_agg.shape[0]} pairs")
        print(f"{'='*60}")

        # 1. Train on aggregated dataset
        print("  Training...", flush=True)
        loss = train_policy(policy, obs_agg, act_agg, vec_norm,
                           epochs=epochs_per_round, batch_size=256, lr=lr)

        # 2. Evaluate
        eval_mean, eval_std = evaluate_policy(ppo_model, vec_norm)
        improved = eval_mean > best_eval
        if improved:
            best_eval = eval_mean
            ppo_model.save('data/models/ppo_gnn_dagger_best')
            with open('data/models/vec_normalize_gnn_dagger.pkl', 'wb') as f:
                pickle.dump(vec_norm, f)

        star = " ★ BEST" if improved else ""
        print(f"  Eval: {eval_mean:+.1f} +/- {eval_std:.1f} | Loss: {loss:.6f}{star}", flush=True)

        # 3. DAgger rollout: collect new data using POLICY's trajectory + Oracle labels
        #    Beta decreases over rounds (start with more Oracle, end with more policy)
        beta = max(0.0, 1.0 - rnd / (n_rounds - 1)) if n_rounds > 1 else 0.0
        rollout_seeds = list(range(rnd * rollouts_per_round, (rnd + 1) * rollouts_per_round))

        # Pick a random subset of 4 scenarios for each round (efficiency)
        rng = np.random.default_rng(rnd)
        round_scenarios = [SCENARIOS[i] for i in rng.choice(len(SCENARIOS), size=4, replace=False)]

        print(f"  DAgger rollout (beta={beta:.2f}, {len(rollout_seeds)} seeds × 4 scenarios)...", flush=True)
        new_obs, new_act = dagger_rollout(ppo_model, vec_norm, rollout_seeds,
                                           scenarios=round_scenarios, beta=beta)
        print(f"  New data: {new_obs.shape[0]} pairs", flush=True)

        # 4. Aggregate
        obs_agg = np.concatenate([obs_agg, new_obs], axis=0)
        act_agg = np.concatenate([act_agg, new_act], axis=0)

    # Final save
    ppo_model.save('data/models/ppo_gnn_dagger')
    with open('data/models/vec_normalize_gnn_dagger_final.pkl', 'wb') as f:
        pickle.dump(vec_norm, f)

    print(f"\n{'='*60}")
    print(f"  DAgger COMPLETE")
    print(f"  Final dataset: {obs_agg.shape[0]} pairs")
    print(f"  Best eval: {best_eval:+.1f}")
    print(f"  Models: ppo_gnn_dagger_best.zip (best), ppo_gnn_dagger.zip (final)")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--rollouts', type=int, default=5, help='Seeds per DAgger round')
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    main(n_rounds=args.rounds, epochs_per_round=args.epochs,
         rollouts_per_round=args.rollouts, lr=args.lr)
