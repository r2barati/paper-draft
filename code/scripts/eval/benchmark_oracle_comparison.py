"""
benchmark_oracle_comparison.py
Compares Old Oracle (one-shot LP / OptimisticEndogenous) vs
Clairvoyant Oracle (rolling-horizon re-planning) across all scenarios.

Usage:
    python -m scripts.eval.benchmark_oracle_comparison [--seeds N]
"""
import os, sys, time
import numpy as np
import pandas as pd
from itertools import product

from scripts.eval.benchmark_engine.runners import run_oracle
from scripts.eval.benchmark_engine.config import (
    NETWORKS, GOODWILL_FLAGS, BACKLOG_FLAGS, PLANNING_HORIZON,
    DEMAND_TYPES, DEMAND_CONFIGS
)
from src.agents.oracle import ClairvoyantOracle
from src.envs.core.environment import CoreEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_SEEDS = 30
SEEDS = list(range(42, 42 + NUM_SEEDS))

RESULTS_DIR = 'data/results'
COMPARISON_CSV = os.path.join(RESULTS_DIR, 'oracle_comparison_per_seed.csv')
SUMMARY_CSV = os.path.join(RESULTS_DIR, 'oracle_comparison_summary.csv')

def compute_kpi(env, H):
    """Compute KPIs from a finished environment."""
    profit = float(np.sum(env.P))
    total_D = float(np.sum(env.D))
    total_U = float(np.sum(env.U))
    fill_rate = max(0.0, 1 - total_U / total_D) if total_D > 0 else 1.0
    unfulfilled = total_U
    # Average inventory across all main nodes over all periods
    main_node_idxs = [env.network.node_map[n] for n in env.network.graph.nodes()
                      if n not in env.network.market and n not in env.network.rawmat]
    avg_inv = float(np.mean(env.X[:H, main_node_idxs]))
    avg_backlog = float(np.mean(env.U[:H]))
    sentiment = float(env.demand_engine.sentiment)
    return {
        'profit': profit,
        'fill_rate': fill_rate,
        'unfulfilled': unfulfilled,
        'avg_inv': avg_inv,
        'avg_backlog': avg_backlog,
        'final_sentiment': sentiment,
    }

def run_clairvoyant(env_kwargs, seed, planning_horizon=30):
    """Run ClairvoyantOracle and return KPI dict."""
    co = ClairvoyantOracle(env_kwargs, planning_horizon=planning_horizon)
    t0 = time.time()
    actions, env = co.solve(seed)
    solve_time = time.time() - t0
    kpis = compute_kpi(env, planning_horizon)
    kpis['solve_time'] = solve_time
    return kpis

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Build scenario list
    scenarios = []
    for network in NETWORKS:
        for demand_label in DEMAND_TYPES:
            demand_base_cfg = DEMAND_CONFIGS[demand_label].copy()
            for use_goodwill in GOODWILL_FLAGS:
                for backlog in BACKLOG_FLAGS:
                    demand_cfg = demand_base_cfg.copy()
                    demand_cfg['use_goodwill'] = use_goodwill
                    scenarios.append({
                        'network': network,
                        'demand_label': demand_label,
                        'use_goodwill': use_goodwill,
                        'backlog': backlog,
                        'demand_config': demand_cfg,
                    })
    
    total = len(scenarios)
    print(f"{'='*80}")
    print(f"Oracle Comparison: Old (One-Shot) vs Clairvoyant (Rolling-Horizon)")
    print(f"  Scenarios: {total}")
    print(f"  Seeds per scenario: {NUM_SEEDS}")
    print(f"  Total solves: {total * NUM_SEEDS * 2}")
    print(f"{'='*80}")
    
    rows = []
    
    for idx, sc in enumerate(scenarios, 1):
        env_kwargs = {
            'scenario': sc['network'],
            'backlog': sc['backlog'],
            'demand_config': sc['demand_config'],
            'num_periods': PLANNING_HORIZON,
        }
        
        label = f"{sc['network']} | {sc['demand_label']} | GW={sc['use_goodwill']} | BL={sc['backlog']}"
        print(f"\n[{idx}/{total}] {label}")
        
        t0_scenario = time.time()
        for seed in SEEDS:
            # --- Old Oracle ---
            old = run_oracle(env_kwargs, seed=seed, planning_horizon=PLANNING_HORIZON)
            
            # --- Clairvoyant Oracle ---
            clv = run_clairvoyant(env_kwargs, seed=seed, planning_horizon=PLANNING_HORIZON)
            
            rows.append({
                'Network': sc['network'],
                'Demand': sc['demand_label'],
                'Goodwill': sc['use_goodwill'],
                'Backlog': sc['backlog'],
                'Seed': seed,
                # Old Oracle
                'Old_Profit': old['profit'],
                'Old_FillRate': old['fill_rate'],
                'Old_Unfulfilled': old['unfulfilled'],
                'Old_AvgInv': old['avg_inv'],
                'Old_AvgBacklog': old['avg_backlog'],
                'Old_Sentiment': old.get('final_sentiment', 1.0),
                'Old_SolveTime': old.get('solve_time', 0),
                # Clairvoyant Oracle
                'Clv_Profit': clv['profit'],
                'Clv_FillRate': clv['fill_rate'],
                'Clv_Unfulfilled': clv['unfulfilled'],
                'Clv_AvgInv': clv['avg_inv'],
                'Clv_AvgBacklog': clv['avg_backlog'],
                'Clv_Sentiment': clv['final_sentiment'],
                'Clv_SolveTime': clv['solve_time'],
            })
        
        elapsed = time.time() - t0_scenario
        # Quick summary for this scenario
        sc_rows = [r for r in rows if r['Network']==sc['network'] and r['Demand']==sc['demand_label'] 
                   and r['Goodwill']==sc['use_goodwill'] and r['Backlog']==sc['backlog']]
        old_mean = np.mean([r['Old_Profit'] for r in sc_rows])
        clv_mean = np.mean([r['Clv_Profit'] for r in sc_rows])
        diff = clv_mean - old_mean
        print(f"  Old: {old_mean:.1f} | Clv: {clv_mean:.1f} | Δ={diff:+.1f} | {elapsed:.1f}s")
    
    # Save per-seed CSV
    df = pd.DataFrame(rows)
    df.to_csv(COMPARISON_CSV, index=False)
    print(f"\nPer-seed results saved to {COMPARISON_CSV}")
    
    # Summary
    summary = df.groupby(['Network', 'Demand', 'Goodwill', 'Backlog']).agg(
        N=('Old_Profit', 'count'),
        Old_Profit_Mean=('Old_Profit', 'mean'),
        Old_Profit_Std=('Old_Profit', 'std'),
        Old_FillRate_Mean=('Old_FillRate', 'mean'),
        Clv_Profit_Mean=('Clv_Profit', 'mean'),
        Clv_Profit_Std=('Clv_Profit', 'std'),
        Clv_FillRate_Mean=('Clv_FillRate', 'mean'),
        Clv_Sentiment_Mean=('Clv_Sentiment', 'mean'),
        Clv_AvgTime=('Clv_SolveTime', 'mean'),
    ).reset_index()
    
    summary['Profit_Diff'] = summary['Clv_Profit_Mean'] - summary['Old_Profit_Mean']
    summary['Profit_Diff_Pct'] = (summary['Profit_Diff'] / summary['Old_Profit_Mean'].abs()) * 100
    summary['FillRate_Diff'] = summary['Clv_FillRate_Mean'] - summary['Old_FillRate_Mean']
    
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Summary saved to {SUMMARY_CSV}")
    
    # Print summary table
    print(f"\n{'='*100}")
    print("COMPARISON SUMMARY")
    print(f"{'='*100}")
    print(summary[['Network','Demand','Goodwill','Backlog','Old_Profit_Mean','Clv_Profit_Mean',
                    'Profit_Diff','Profit_Diff_Pct','Old_FillRate_Mean','Clv_FillRate_Mean',
                    'FillRate_Diff']].to_string(index=False))

if __name__ == '__main__':
    main()
