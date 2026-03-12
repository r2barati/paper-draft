#!/usr/bin/env python3
"""
train_gnn_dagger_specialist.py — Specialist DAgger: one model per demand type.

Trains 4 DAgger models (stationary, trend, seasonal, shock), each with:
  - Demos from only that demand type (4 scenarios: ±goodwill × ±backlog)
  - DAgger rollouts restricted to that demand type
  - Eval on that demand type

This fixes the generalist DAgger's cross-scenario confusion.
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


def get_scenarios(demand_type):
    return list(itertools.product([demand_type], [False, True], [False, True]))


def make_env(demand_type='stationary'):
    def _init():
        env = CoreEnv(scenario='base', num_periods=H, backlog=True,
                      demand_config={'type': demand_type, 'use_goodwill': False, 'base_mu': 20})
        env = DomainFeatureWrapper(env, is_blind=False, enhanced=True, grouped=True)
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        env = Monitor(env)
        return env
    return _init


def collect_demos(scenarios, seeds):
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
                raw = oracle_actions[t]
                lo, hi = env.unwrapped.action_space.low, env.unwrapped.action_space.high
                rescaled = np.clip(2.0 * (raw - lo) / (hi - lo + 1e-8) - 1.0, -1, 1)
                all_obs.append(obs.astype(np.float32))
                all_actions.append(rescaled.astype(np.float32))
                obs, _, done, trunc, _ = env.step(raw)
                if done or trunc: break
    return np.stack(all_obs), np.stack(all_actions)


def dagger_rollout(model, vec_norm, scenarios, seeds, beta=0.0):
    all_obs, all_actions = [], []
    for dem, gw, bl in scenarios:
        env_kwargs = {'scenario': 'base', 'backlog': bl, 'num_periods': H,
                      'demand_config': {'type': dem, 'use_goodwill': gw, 'base_mu': 20}}
        for seed in seeds:
            oracle = ClairvoyantOracle(env_kwargs, planning_horizon=H)
            oracle_actions, _ = oracle.solve(seed=seed)
            env = CoreEnv(**env_kwargs)
            env = DomainFeatureWrapper(env, is_blind=False, enhanced=True, grouped=True)
            env_r = RescaleAction(env, min_action=-1.0, max_action=1.0)
            obs, _ = env_r.reset(seed=seed)
            for t in range(H):
                all_obs.append(obs.astype(np.float32))
                raw = oracle_actions[t]
                lo, hi = env.unwrapped.action_space.low, env.unwrapped.action_space.high
                rescaled = np.clip(2.0 * (raw - lo) / (hi - lo + 1e-8) - 1.0, -1, 1)
                all_actions.append(rescaled.astype(np.float32))
                if np.random.random() < beta:
                    obs, _, done, trunc, _ = env_r.step(rescaled)
                else:
                    obs_n = vec_norm.normalize_obs(obs.reshape(1, -1)).squeeze(0)
                    act, _ = model.predict(obs_n, deterministic=True)
                    obs, _, done, trunc, _ = env_r.step(act)
                if done or trunc: break
    return np.stack(all_obs), np.stack(all_actions)


def train_policy(policy, obs, act, vec_norm, epochs=50, bs=256, lr=1e-3):
    device = policy.device
    obs_n = np.array([vec_norm.normalize_obs(obs[i:i+1]).squeeze(0)
                      for i in range(obs.shape[0])], dtype=np.float32)
    loader = DataLoader(TensorDataset(
        torch.tensor(obs_n, dtype=torch.float32, device=device),
        torch.tensor(act, dtype=torch.float32, device=device)
    ), batch_size=bs, shuffle=True)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    crit = nn.MSELoss()
    final_loss = 0
    for ep in range(epochs):
        tot, n = 0, 0
        for ob, ac in loader:
            feat = policy.extract_features(ob, policy.features_extractor)
            lat = policy.mlp_extractor.forward_actor(feat)
            pred = policy.action_net(lat)
            loss = crit(pred, ac)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        final_loss = tot / n
        if (ep + 1) % 10 == 0:
            print(f"      Epoch {ep+1:3d}: MSE={final_loss:.6f}", flush=True)
    return final_loss


def evaluate(model, vec_norm, demand_type, seeds=[42, 123, 456]):
    profits = []
    for bl in [True, False]:
        env_kwargs = {'scenario': 'base', 'backlog': bl, 'num_periods': H,
                      'demand_config': {'type': demand_type, 'use_goodwill': False, 'base_mu': 20}}
        for seed in seeds:
            env = CoreEnv(**env_kwargs)
            env = DomainFeatureWrapper(env, is_blind=False, enhanced=True, grouped=True)
            env = RescaleAction(env, min_action=-1.0, max_action=1.0)
            obs, _ = env.reset(seed=seed)
            r = 0
            for t in range(H):
                obs_n = vec_norm.normalize_obs(obs.reshape(1, -1)).squeeze(0)
                act, _ = model.predict(obs_n, deterministic=True)
                obs, rew, done, trunc, info = env.step(act)
                r += info.get('raw_reward', rew)
                if done or trunc: break
            profits.append(r)
    return np.mean(profits), np.std(profits)


def train_specialist(demand_type, n_rounds=10, epochs=50, lr=1e-3):
    print(f"\n{'='*60}")
    print(f"  SPECIALIST DAgger: {demand_type.upper()}")
    print(f"{'='*60}")

    scenarios = get_scenarios(demand_type)
    vec_env = DummyVecEnv([make_env(demand_type)])
    vec_norm = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.995)

    policy_kwargs = dict(
        features_extractor_class=GNNFeaturesExtractorV3,
        features_extractor_kwargs=dict(features_dim=256, scenario='base', hidden_dim=64, n_layers=3),
        net_arch=[256, 128],
    )
    model = PPO('MlpPolicy', vec_env, policy_kwargs=policy_kwargs, learning_rate=1e-4, verbose=0)

    # Phase 0: initial demos
    print(f"  Collecting {demand_type} Oracle demos...", flush=True)
    obs_agg, act_agg = collect_demos(scenarios, seeds=list(range(5)))
    print(f"  Initial: {obs_agg.shape[0]} pairs")

    for i in range(min(2000, obs_agg.shape[0])):
        vec_norm.normalize_obs(obs_agg[i:i+1])

    best_eval = -np.inf
    for rnd in range(n_rounds):
        print(f"  Round {rnd+1}/{n_rounds} ({obs_agg.shape[0]} pairs):", flush=True)
        train_policy(model.policy, obs_agg, act_agg, vec_norm, epochs=epochs, lr=lr)

        ev_mean, ev_std = evaluate(model, vec_norm, demand_type)
        improved = ev_mean > best_eval
        if improved:
            best_eval = ev_mean
            model.save(f'data/models/ppo_dagger_{demand_type}')
            with open(f'data/models/vec_normalize_dagger_{demand_type}.pkl', 'wb') as f:
                pickle.dump(vec_norm, f)

        star = " ★" if improved else ""
        print(f"    Eval: {ev_mean:+.1f} +/- {ev_std:.1f}{star}", flush=True)

        beta = max(0.0, 1.0 - rnd / (n_rounds - 1)) if n_rounds > 1 else 0.0
        rollout_seeds = list(range(rnd * 5, (rnd + 1) * 5))
        new_obs, new_act = dagger_rollout(model, vec_norm, scenarios, rollout_seeds, beta=beta)
        obs_agg = np.concatenate([obs_agg, new_obs])
        act_agg = np.concatenate([act_agg, new_act])

    print(f"  ✅ {demand_type} DONE — Best: {best_eval:+.1f}")
    vec_env.close()
    return best_eval


def main():
    print("=" * 60)
    print("  SPECIALIST DAgger — 4 Demand-Type-Specific Models")
    print("=" * 60)

    results = {}
    for dt in ['stationary', 'trend', 'seasonal', 'shock']:
        results[dt] = train_specialist(dt, n_rounds=8, epochs=50, lr=1e-3)

    print("\n" + "=" * 60)
    print("  ALL SPECIALISTS COMPLETE")
    for dt, val in results.items():
        print(f"    {dt:12s}: Best eval = {val:+.1f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
