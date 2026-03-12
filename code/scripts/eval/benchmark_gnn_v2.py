#!/usr/bin/env python3
"""
benchmark_gnn_v2.py — Benchmark GNN V2 across all scenarios and compare to Oracle/MSSP

Runs the trained GNN V2 model on all 32 scenario configurations:
  2 networks × 4 demand types × 2 goodwill × 2 backlog = 32 scenarios
  × 5 seeds each = 160 evaluations

Outputs a comparison table against cached Oracle/MSSP baselines.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import itertools
import pickle
import time
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from gymnasium.wrappers import RescaleAction

from src.envs.core.environment import CoreEnv
from src.envs.wrappers.feature_wrappers import DomainFeatureWrapper
from src.models.gnn_extractor_v2 import GNNFeaturesExtractorV2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = 'data/models/ppo_gnn_v2'
STATS_PATH = 'data/models/vec_normalize_gnn_v2.pkl'

# Use best model from eval callback if available
BEST_MODEL_PATH = 'ppo_gnn_v2_logs/best_model'

BASELINE_CACHE = 'data/results/cache/benchmark_baselines_cache.csv'
OUTPUT_CSV = 'data/results/benchmark_gnn_v2_results.csv'

NETWORKS = ['base']  # GNN V2 trained on base only
DEMAND_TYPES = ['stationary', 'trend', 'seasonal', 'shock']
GOODWILL_FLAGS = [False, True]
BACKLOG_FLAGS = [False, True]
SEEDS = [42, 123, 456, 789, 1010]
PLANNING_HORIZON = 30

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kpi(env, episode_reward):
    avg_inv = np.mean([np.sum(env.X[t]) for t in range(env.period)]) if env.period > 0 else 0
    total_u = np.sum(env.U)
    total_d = np.sum(env.D)
    fill_rate = max(0.0, 1.0 - (total_u / total_d)) if total_d > 0 else 1.0
    avg_backlog = np.mean(env.U) if env.U.size > 0 else 0
    final_sentiment = getattr(env.demand_engine, 'sentiment', 1.0)
    return {
        "profit": episode_reward,
        "avg_inv": avg_inv,
        "unfulfilled": total_u,
        "fill_rate": fill_rate,
        "avg_backlog": avg_backlog,
        "final_sentiment": final_sentiment,
    }

def load_model(model_path, stats_path):
    """Load GNN V2 model and normalizer."""
    # Try best model first, fallback to final
    for mp in [BEST_MODEL_PATH, model_path]:
        zp = mp if mp.endswith('.zip') else mp + '.zip'
        if os.path.exists(zp):
            print(f"  Loading model from: {zp}")
            model = PPO.load(mp, custom_objects={"GNNFeaturesExtractorV2": GNNFeaturesExtractorV2})
            break
    else:
        raise FileNotFoundError(f"No model found at {model_path} or {BEST_MODEL_PATH}")

    with open(stats_path, 'rb') as f:
        norm_env = pickle.load(f)
    norm_env.training = False
    norm_env.norm_reward = False
    return model, norm_env

def run_gnn_v2(env_kwargs, seed, planning_horizon, model, norm_env):
    """Run one episode with GNN V2."""
    env = CoreEnv(**env_kwargs)
    env = DomainFeatureWrapper(env, is_blind=False, enhanced=True)  # V2 features!
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)

    obs, _ = env.reset(seed=seed)
    episode_reward = 0
    for t in range(planning_horizon):
        obs_2d = obs.reshape(1, -1)
        norm_obs = norm_env.normalize_obs(obs_2d).squeeze(0)
        action, _ = model.predict(norm_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += info.get('raw_reward', reward)
        if terminated or truncated:
            break

    return _kpi(env.unwrapped, episode_reward)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  GNN V2 Comprehensive Benchmark")
    print("=" * 70)

    # Load model once
    model, norm_env = load_model(MODEL_PATH, STATS_PATH)

    # Generate all scenarios
    scenarios = list(itertools.product(NETWORKS, DEMAND_TYPES, GOODWILL_FLAGS, BACKLOG_FLAGS))
    print(f"\n  {len(scenarios)} scenario configurations × {len(SEEDS)} seeds = {len(scenarios) * len(SEEDS)} evaluations\n")

    results = []
    for i, (network, demand, goodwill, backlog) in enumerate(scenarios):
        label = f"{network}/{demand}/G={goodwill}/B={backlog}"
        profits = []

        for seed in SEEDS:
            env_kwargs = {
                'scenario': network,
                'backlog': backlog,
                'num_periods': PLANNING_HORIZON,
                'demand_config': {
                    'type': demand,
                    'use_goodwill': goodwill,
                    'base_mu': 20,
                },
            }
            try:
                kpi = run_gnn_v2(env_kwargs, seed, PLANNING_HORIZON, model, norm_env)
                profits.append(kpi['profit'])
                results.append({
                    'Network': network,
                    'Demand': demand,
                    'Goodwill': goodwill,
                    'Backlog': backlog,
                    'Seed': seed,
                    **kpi,
                })
            except Exception as e:
                print(f"  ❌ FAILED: {label} seed={seed}: {e}")

        mean_p = np.mean(profits) if profits else 0
        std_p = np.std(profits) if profits else 0
        print(f"  [{i+1:2d}/{len(scenarios)}] {label:45s}  profit={mean_p:+8.1f} ± {std_p:.1f}")

    # Save raw results
    df_gnn = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_gnn.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  📄 Raw results saved to {OUTPUT_CSV}")

    # Aggregate by scenario
    agg = df_gnn.groupby(['Network', 'Demand', 'Goodwill', 'Backlog']).agg(
        GNN_V2_Profit=('profit', 'mean'),
        GNN_V2_Std=('profit', 'std'),
        GNN_V2_FillRate=('fill_rate', 'mean'),
    ).reset_index()

    # Load and merge baselines
    if os.path.exists(BASELINE_CACHE):
        df_base = pd.read_csv(BASELINE_CACHE)
        merged = agg.merge(
            df_base[['Network', 'Demand', 'Goodwill', 'Backlog',
                      'Oracle_Profit', 'MSSP_Profit', 'DLP_Profit', 'Heuristic_Profit',
                      'Oracle_FillRate', 'MSSP_FillRate']],
            on=['Network', 'Demand', 'Goodwill', 'Backlog'],
            how='left',
        )
        merged['Pct_Oracle'] = (merged['GNN_V2_Profit'] / merged['Oracle_Profit'] * 100).round(1)
        merged['Pct_MSSP'] = (merged['GNN_V2_Profit'] / merged['MSSP_Profit'] * 100).round(1)
        merged['Gap_Oracle'] = (merged['Oracle_Profit'] - merged['GNN_V2_Profit']).round(1)
        merged['Gap_MSSP'] = (merged['MSSP_Profit'] - merged['GNN_V2_Profit']).round(1)
        merged['Beats_MSSP'] = merged['GNN_V2_Profit'] > merged['MSSP_Profit']
        merged['Beats_DLP'] = merged['GNN_V2_Profit'] > merged['DLP_Profit']
        merged['Beats_Heuristic'] = merged['GNN_V2_Profit'] > merged['Heuristic_Profit']
    else:
        merged = agg
        print("  ⚠️ No baseline cache found — skipping comparison.")

    # Print comparison table
    print(f"\n{'=' * 100}")
    print("  GNN V2 vs Oracle / MSSP — All Scenarios")
    print(f"{'=' * 100}\n")

    display_cols = ['Network', 'Demand', 'Goodwill', 'Backlog',
                    'GNN_V2_Profit', 'Oracle_Profit', 'MSSP_Profit',
                    'Pct_Oracle', 'Pct_MSSP', 'Beats_MSSP']
    available = [c for c in display_cols if c in merged.columns]
    print(merged[available].round(1).to_string(index=False))

    # Summary statistics
    if 'Pct_Oracle' in merged.columns:
        print(f"\n{'─' * 60}")
        print(f"  SUMMARY STATISTICS")
        print(f"{'─' * 60}")
        valid = merged.dropna(subset=['Pct_Oracle'])
        print(f"  Mean % of Oracle: {valid['Pct_Oracle'].mean():.1f}%")
        print(f"  Min  % of Oracle: {valid['Pct_Oracle'].min():.1f}% ({valid.loc[valid['Pct_Oracle'].idxmin(), 'Demand']})")
        print(f"  Max  % of Oracle: {valid['Pct_Oracle'].max():.1f}% ({valid.loc[valid['Pct_Oracle'].idxmax(), 'Demand']})")
        print(f"  Scenarios beating MSSP:      {valid['Beats_MSSP'].sum()}/{len(valid)}")
        print(f"  Scenarios beating DLP:        {valid['Beats_DLP'].sum()}/{len(valid)}")
        print(f"  Scenarios beating Heuristic:  {valid['Beats_Heuristic'].sum()}/{len(valid)}")

    # Save final comparison
    comp_csv = 'data/results/benchmark_gnn_v2_comparison.csv'
    merged.to_csv(comp_csv, index=False)
    print(f"\n  📄 Comparison table saved to {comp_csv}")

if __name__ == '__main__':
    start = time.perf_counter()
    main()
    elapsed = time.perf_counter() - start
    print(f"\n  ⏱️ Total benchmark time: {elapsed:.1f}s")
