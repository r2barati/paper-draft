"""
benchmark_oracle_comprehensive.py — Comprehensive Oracle Assessment

Evaluates the Oracle (perfect-information LP) across ALL possible demand
configurations, including composable effects, at 30 seeds per scenario.

Demand Configs (8 total):
  1. stationary         []
  2. trend              ['trend']
  3. seasonal           ['seasonal']
  4. shock              ['shock']
  5. trend+seasonal     ['trend', 'seasonal']
  6. trend+shock        ['trend', 'shock']
  7. seasonal+shock     ['seasonal', 'shock']
  8. trend+seasonal+shock ['trend', 'seasonal', 'shock']

Scenario Matrix:
  8 demands × 2 networks × 2 goodwill × 2 backlog = 64 scenarios
  64 scenarios × 30 seeds = 1,920 Oracle solves

Output:
  - data/results/oracle_comprehensive_per_seed.csv   (1,920 rows)
  - data/results/oracle_comprehensive_summary.csv    (64 rows)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
import numpy as np
import pandas as pd
from itertools import product

from scripts.eval.benchmark_engine.runners import run_oracle
from scripts.eval.benchmark_engine.config import (
    NETWORKS, GOODWILL_FLAGS, BACKLOG_FLAGS, PLANNING_HORIZON,
    DEMAND_TYPES, DEMAND_CONFIGS
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_SEEDS = 30
SEEDS = list(range(42, 42 + NUM_SEEDS))

RESULTS_DIR = 'data/results'
PER_SEED_CSV = os.path.join(RESULTS_DIR, 'oracle_comprehensive_per_seed.csv')
SUMMARY_CSV = os.path.join(RESULTS_DIR, 'oracle_comprehensive_summary.csv')


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

    total_scenarios = len(scenarios)
    total_solves = total_scenarios * NUM_SEEDS
    print("=" * 80)
    print(f"Comprehensive Oracle Assessment")
    print(f"  Scenarios: {total_scenarios}")
    print(f"  Seeds per scenario: {NUM_SEEDS}")
    print(f"  Total Oracle solves: {total_solves}")
    print("=" * 80)

    all_rows = []
    global_start = time.time()

    for s_idx, scen in enumerate(scenarios):
        tag = f"{scen['network']} | {scen['demand_label']} | GW={scen['use_goodwill']} | BL={scen['backlog']}"
        print(f"\n[{s_idx + 1}/{total_scenarios}] {tag}")

        env_kwargs = {
            'scenario': scen['network'],
            'backlog': scen['backlog'],
            'demand_config': scen['demand_config'],
            'num_periods': PLANNING_HORIZON,
        }

        seed_start = time.time()
        for seed in SEEDS:
            t0 = time.time()
            try:
                result = run_oracle(env_kwargs, seed, PLANNING_HORIZON)
            except Exception as e:
                print(f"  ERROR seed={seed}: {e}")
                result = None

            elapsed = time.time() - t0

            row = {
                'Network': scen['network'],
                'Demand': scen['demand_label'],
                'Goodwill': scen['use_goodwill'],
                'Backlog': scen['backlog'],
                'Seed': seed,
                'Oracle_Profit': result['profit'] if result else np.nan,
                'Oracle_FillRate': result['fill_rate'] if result else np.nan,
                'Oracle_AvgInv': result['avg_inv'] if result else np.nan,
                'Oracle_Unfulfilled': result['unfulfilled'] if result else np.nan,
                'Oracle_AvgBacklog': result['avg_backlog'] if result else np.nan,
                'Oracle_FinalSentiment': result['final_sentiment'] if result else np.nan,
                'Oracle_SolveTime_s': elapsed,
            }
            all_rows.append(row)

        seed_elapsed = time.time() - seed_start
        # Quick stats for this scenario
        profits = [r['Oracle_Profit'] for r in all_rows[-NUM_SEEDS:] if not np.isnan(r['Oracle_Profit'])]
        if profits:
            print(f"  Profit: mean={np.mean(profits):.2f} ± {np.std(profits):.2f}  |  "
                  f"Fill: {np.mean([r['Oracle_FillRate'] for r in all_rows[-NUM_SEEDS:]]):.4f}  |  "
                  f"Time: {seed_elapsed:.1f}s ({seed_elapsed/NUM_SEEDS:.1f}s/seed)")
        else:
            print(f"  ALL FAILED")

    total_elapsed = time.time() - global_start
    print(f"\n{'=' * 80}")
    print(f"Done! Total time: {total_elapsed/60:.1f} min")

    # Save per-seed CSV
    df_per_seed = pd.DataFrame(all_rows)
    df_per_seed.to_csv(PER_SEED_CSV, index=False)
    print(f"Per-seed results saved to: {PER_SEED_CSV}")

    # Build summary
    group_cols = ['Network', 'Demand', 'Goodwill', 'Backlog']
    kpi_cols = ['Oracle_Profit', 'Oracle_FillRate', 'Oracle_AvgInv',
                'Oracle_Unfulfilled', 'Oracle_AvgBacklog', 'Oracle_FinalSentiment']

    summary_rows = []
    for keys, grp in df_per_seed.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row['Num_Seeds'] = len(grp)
        for col in kpi_cols:
            vals = grp[col].dropna()
            row[f'{col}_Mean'] = vals.mean()
            row[f'{col}_Std'] = vals.std()
            row[f'{col}_Min'] = vals.min()
            row[f'{col}_Max'] = vals.max()
        row['Avg_SolveTime_s'] = grp['Oracle_SolveTime_s'].mean()
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Summary results saved to: {SUMMARY_CSV}")

    # Print summary table
    print(f"\n{'=' * 80}")
    print("ORACLE COMPREHENSIVE SUMMARY")
    print(f"{'=' * 80}")
    display_cols = ['Network', 'Demand', 'Goodwill', 'Backlog',
                    'Oracle_Profit_Mean', 'Oracle_Profit_Std',
                    'Oracle_FillRate_Mean', 'Oracle_Unfulfilled_Mean']
    print(df_summary[display_cols].to_string(index=False))


if __name__ == '__main__':
    main()
