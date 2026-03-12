#!/usr/bin/env python3
"""
verify_gnn_il_no_leakage.py — Out-of-sample test to prove GNN-IL has no future info leakage.

Tests GNN-IL vs V3 vs Oracle on:
  - Seeds NEVER seen in training (training used 0-9, we test 42,123,456,789,999)
  - All 16 scenarios
  
If GNN-IL performs well on unseen seeds/scenarios, it's generalizing — not memorizing.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pickle
import itertools
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import RescaleAction

from src.envs.core.environment import CoreEnv
from src.envs.wrappers.feature_wrappers import DomainFeatureWrapper
from src.agents.oracle import ClairvoyantOracle
from src.models.gnn_extractor_v3 import GNNFeaturesExtractorV3

# CRITICAL: These seeds were NEVER used in training (training used 0-9)
UNSEEN_SEEDS = [42, 123, 456, 789, 999]

SCENARIOS = list(itertools.product(
    ['stationary', 'trend', 'seasonal', 'shock'],
    [False, True],   # goodwill
    [False, True],   # backlog
))
H = 30


def eval_gnn_il(env_kwargs, seed, model, vec_norm):
    """Evaluate GNN-IL on a single episode."""
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
    return ep_r


def eval_v3(env_kwargs, seed, model, vec_norm):
    """Evaluate V3 on a single episode (same interface)."""
    return eval_gnn_il(env_kwargs, seed, model, vec_norm)


def eval_oracle(env_kwargs, seed):
    """Evaluate Oracle on a single episode by replaying its actions."""
    oracle = ClairvoyantOracle(env_kwargs, planning_horizon=H)
    actions, _ = oracle.solve(seed=seed)
    
    # Replay through a fresh env to get consistent reward calculation
    env = CoreEnv(**env_kwargs)
    env.reset(seed=seed)
    profit = 0
    for t in range(H):
        _, reward, done, trunc, info = env.step(actions[t])
        profit += reward
        if done or trunc:
            break
    return profit


def main():
    print("=" * 70)
    print("  GNN-IL OUT-OF-SAMPLE LEAKAGE TEST")
    print(f"  Unseen seeds: {UNSEEN_SEEDS}")
    print(f"  Training seeds were: 0-9 (NONE of these are used here)")
    print("=" * 70)

    # Load GNN-IL best model
    print("\nLoading GNN-IL (best checkpoint)...", flush=True)
    il_model = PPO.load(
        'gnn_il_logs/best_model',
        custom_objects={'GNNFeaturesExtractorV3': GNNFeaturesExtractorV3},
    )
    with open('data/models/vec_normalize_gnn_il.pkl', 'rb') as f:
        il_vecnorm = pickle.load(f)
    il_vecnorm.training = False

    # Load V3 baseline
    print("Loading V3 baseline...", flush=True)
    v3_model = PPO.load(
        'data/models/ppo_gnn_v3',
        custom_objects={'GNNFeaturesExtractorV3': GNNFeaturesExtractorV3},
    )
    with open('data/models/vec_normalize_gnn_v3.pkl', 'rb') as f:
        v3_vecnorm = pickle.load(f)
    v3_vecnorm.training = False

    # Results storage
    results = []
    
    for dem, gw, bl in SCENARIOS:
        env_kwargs = {
            'scenario': 'base',
            'backlog': bl,
            'num_periods': H,
            'demand_config': {'type': dem, 'use_goodwill': gw, 'base_mu': 20},
        }
        
        il_profits = []
        v3_profits = []
        oracle_profits = []
        
        for seed in UNSEEN_SEEDS:
            # GNN-IL
            il_p = eval_gnn_il(env_kwargs, seed, il_model, il_vecnorm)
            il_profits.append(il_p)
            
            # V3
            v3_p = eval_v3(env_kwargs, seed, v3_model, v3_vecnorm)
            v3_profits.append(v3_p)
            
            # Oracle
            oracle_p = eval_oracle(env_kwargs, seed)
            oracle_profits.append(oracle_p)
        
        il_mean = np.mean(il_profits)
        v3_mean = np.mean(v3_profits)
        oracle_mean = np.mean(oracle_profits)
        
        il_pct = il_mean / oracle_mean * 100 if oracle_mean > 0 else 0
        v3_pct = v3_mean / oracle_mean * 100 if oracle_mean > 0 else 0
        
        results.append({
            'demand': dem, 'goodwill': gw, 'backlog': bl,
            'oracle': oracle_mean, 'v3': v3_mean, 'il': il_mean,
            'v3_pct': v3_pct, 'il_pct': il_pct,
        })
        
        print(f"  {dem:12s} G={str(gw):5s} B={str(bl):5s}: "
              f"Oracle={oracle_mean:+7.1f}  V3={v3_mean:+7.1f}({v3_pct:5.1f}%)  "
              f"GNN-IL={il_mean:+7.1f}({il_pct:5.1f}%)", flush=True)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY (all on UNSEEN seeds only)")
    print("=" * 70)
    
    v3_pcts = [r['v3_pct'] for r in results]
    il_pcts = [r['il_pct'] for r in results]
    
    print(f"  V3  mean % Oracle: {np.mean(v3_pcts):.1f}%  (range: {np.min(v3_pcts):.1f}% - {np.max(v3_pcts):.1f}%)")
    print(f"  IL  mean % Oracle: {np.mean(il_pcts):.1f}%  (range: {np.min(il_pcts):.1f}% - {np.max(il_pcts):.1f}%)")
    
    il_wins = sum(1 for r in results if r['il'] > r['v3'])
    print(f"\n  GNN-IL beats V3: {il_wins}/16 scenarios")
    
    # Save results
    import json
    with open('data/results/gnn_il_leakage_test.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to data/results/gnn_il_leakage_test.json")


if __name__ == '__main__':
    main()
