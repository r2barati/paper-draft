#!/usr/bin/env python3
"""
benchmark_dlp_full.py — DLP + Oracle benchmark across all 64 demand scenarios.

Runs DLP (informed), DLP_Blind, and Oracle across the full combinatorial matrix:
  8 demand types × 2 networks × 2 goodwill × 2 backlog = 64 scenarios
  Each scenario averaged over N seeds (default 5).

Demand types include both individual and composable effects:
  - stationary, trend, seasonal, shock  (individual)
  - trend+seasonal, trend+shock, seasonal+shock, trend+seasonal+shock  (composable)

Output: data/results/dlp_oracle_full_64.csv
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.eval.benchmark_engine.runners import run_dlp, run_oracle, compute_kpi

# ---------------------------------------------------------------------------
# Scenario Matrix
# ---------------------------------------------------------------------------
ALL_NETWORKS = ['base', 'serial']

# Demand configs: label → demand_config dict (without goodwill, added later)
ALL_DEMANDS = {
    'stationary':              {'type': 'stationary'},
    'trend':                   {'type': 'trend'},
    'seasonal':                {'type': 'seasonal'},
    'shock':                   {'type': 'shock'},
    'trend+seasonal':          {'effects': ['trend', 'seasonal']},
    'trend+shock':             {'effects': ['trend', 'shock']},
    'seasonal+shock':          {'effects': ['seasonal', 'shock']},
    'trend+seasonal+shock':    {'effects': ['trend', 'seasonal', 'shock']},
}

GOODWILL_FLAGS = [False, True]
BACKLOG_FLAGS  = [True, False]
PLANNING_HORIZON = 30
DEFAULT_SEEDS = list(range(100, 105))  # 5 seeds

AGENTS = ['Oracle', 'DLP', 'DLP_Blind']

RESULTS_DIR = 'data/results'
OUTPUT_FILE = os.path.join(RESULTS_DIR, 'dlp_oracle_full_64.csv')


def build_env_kwargs(network, demand_label, demand_base_config, use_goodwill, backlog):
    """Build the env_kwargs dict for a single scenario."""
    demand_config = dict(demand_base_config)  # shallow copy
    demand_config['use_goodwill'] = use_goodwill
    demand_config['base_mu'] = 20

    return {
        'scenario': network,
        'num_periods': PLANNING_HORIZON,
        'backlog': backlog,
        'demand_config': demand_config,
    }


def run_scenario(env_kwargs, seed, agent_name):
    """Run a single agent on a single scenario+seed. Returns KPI dict or None."""
    try:
        if agent_name == 'Oracle':
            return run_oracle(env_kwargs, seed, PLANNING_HORIZON)
        elif agent_name == 'DLP':
            return run_dlp(env_kwargs, seed, PLANNING_HORIZON, is_blind=False)
        elif agent_name == 'DLP_Blind':
            return run_dlp(env_kwargs, seed, PLANNING_HORIZON, is_blind=True)
    except Exception as e:
        print(f"  ⚠️  {agent_name} failed (seed={seed}): {e}")
        return None


def build_scenario_list(networks, demand_labels):
    """Build the full list of (network, demand_label, goodwill, backlog) tuples."""
    scenarios = []
    for net in networks:
        for dlabel in demand_labels:
            for gw in GOODWILL_FLAGS:
                for bl in BACKLOG_FLAGS:
                    scenarios.append((net, dlabel, gw, bl))
    return scenarios


def main():
    parser = argparse.ArgumentParser(description='DLP + Oracle Full 64-Scenario Benchmark')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of evaluation seeds (default: 5)')
    parser.add_argument('--networks', nargs='+', default=None,
                        help='Networks to evaluate (default: all)')
    parser.add_argument('--demands', nargs='+', default=None,
                        help='Demand types to evaluate (default: all)')
    args = parser.parse_args()

    seeds = list(range(100, 100 + args.seeds))
    networks = args.networks if args.networks else ALL_NETWORKS
    demand_labels = args.demands if args.demands else list(ALL_DEMANDS.keys())

    # Validate demand labels
    for d in demand_labels:
        if d not in ALL_DEMANDS:
            print(f"❌ Unknown demand type: '{d}'")
            print(f"   Valid options: {list(ALL_DEMANDS.keys())}")
            sys.exit(1)

    scenarios = build_scenario_list(networks, demand_labels)
    total_runs = len(scenarios) * len(seeds) * len(AGENTS)

    print(f"╔══════════════════════════════════════════════════════════╗")
    print(f"║  DLP + Oracle: Full Scenario Benchmark                  ║")
    print(f"╠══════════════════════════════════════════════════════════╣")
    print(f"║  Networks:    {networks}")
    print(f"║  Demands:     {demand_labels}")
    print(f"║  Goodwill:    {GOODWILL_FLAGS}")
    print(f"║  Backlog:     {BACKLOG_FLAGS}")
    print(f"║  Seeds:       {seeds}")
    print(f"║  Scenarios:   {len(scenarios)}")
    print(f"║  Total runs:  {total_runs}")
    print(f"╚══════════════════════════════════════════════════════════╝")
    print()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    t_start = time.time()

    for idx, (net, dlabel, gw, bl) in enumerate(tqdm(scenarios, desc="Scenarios")):
        demand_base = ALL_DEMANDS[dlabel]
        env_kwargs = build_env_kwargs(net, dlabel, demand_base, gw, bl)

        scenario_tag = f"{net}|{dlabel}|GW={gw}|BL={bl}"
        tqdm.write(f"\n[{idx+1}/{len(scenarios)}] {scenario_tag}")

        for agent_name in AGENTS:
            agent_profits = []
            agent_fill_rates = []
            agent_avg_invs = []
            agent_unfulfilled = []
            agent_times = []

            for seed in seeds:
                t0 = time.time()
                kpi = run_scenario(env_kwargs, seed, agent_name)
                elapsed = time.time() - t0

                if kpi is not None:
                    agent_profits.append(kpi['profit'])
                    agent_fill_rates.append(kpi['fill_rate'])
                    agent_avg_invs.append(kpi['avg_inv'])
                    agent_unfulfilled.append(kpi['unfulfilled'])
                    agent_times.append(elapsed)

            if len(agent_profits) > 0:
                all_results.append({
                    'Network': net,
                    'Demand': dlabel,
                    'Goodwill': gw,
                    'Backlog': bl,
                    'Agent': agent_name,
                    'Profit_Mean': np.mean(agent_profits),
                    'Profit_Std': np.std(agent_profits),
                    'FillRate_Mean': np.mean(agent_fill_rates),
                    'AvgInv_Mean': np.mean(agent_avg_invs),
                    'Unfulfilled_Mean': np.mean(agent_unfulfilled),
                    'Time_Mean_Sec': np.mean(agent_times),
                    'Num_Seeds': len(agent_profits),
                })

                tqdm.write(f"  {agent_name:12s}  profit={np.mean(agent_profits):8.1f} ± {np.std(agent_profits):6.1f}  "
                           f"fill={np.mean(agent_fill_rates):.3f}  t={np.mean(agent_times):.2f}s")
            else:
                tqdm.write(f"  {agent_name:12s}  FAILED (no valid seeds)")

    elapsed_total = time.time() - t_start

    # Save long-form results
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Results saved to {OUTPUT_FILE}")
    print(f"   Total time: {elapsed_total/60:.1f} minutes")

    # ---------------------------------------------------------------------------
    # Generate comparison table: pivot to wide format (DLP vs Oracle per scenario)
    # ---------------------------------------------------------------------------
    if len(df) == 0:
        print("No results to display.")
        return

    pivot = df.pivot_table(
        index=['Network', 'Demand', 'Goodwill', 'Backlog'],
        columns='Agent',
        values=['Profit_Mean', 'FillRate_Mean', 'Unfulfilled_Mean'],
        aggfunc='first'
    )

    # Flatten column names
    pivot.columns = [f'{metric}_{agent}' for metric, agent in pivot.columns]
    pivot = pivot.reset_index()

    # Compute gaps
    if 'Profit_Mean_Oracle' in pivot.columns and 'Profit_Mean_DLP' in pivot.columns:
        pivot['DLP_Gap_vs_Oracle_%'] = (
            (pivot['Profit_Mean_Oracle'] - pivot['Profit_Mean_DLP']) / pivot['Profit_Mean_Oracle'].abs() * 100
        ).round(2)

    if 'Profit_Mean_DLP' in pivot.columns and 'Profit_Mean_DLP_Blind' in pivot.columns:
        pivot['VPF_DLP'] = (pivot['Profit_Mean_DLP'] - pivot['Profit_Mean_DLP_Blind']).round(2)

    # Save wide table
    wide_file = os.path.join(RESULTS_DIR, 'dlp_oracle_comparison_64.csv')
    pivot.to_csv(wide_file, index=False)
    print(f"✅ Comparison table saved to {wide_file}")

    # Print summary table
    print("\n" + "=" * 120)
    print("  DLP vs Oracle — Full 64-Scenario Comparison")
    print("=" * 120)

    display_cols = ['Network', 'Demand', 'Goodwill', 'Backlog']
    for col in ['Profit_Mean_Oracle', 'Profit_Mean_DLP', 'Profit_Mean_DLP_Blind',
                'DLP_Gap_vs_Oracle_%', 'VPF_DLP',
                'FillRate_Mean_Oracle', 'FillRate_Mean_DLP']:
        if col in pivot.columns:
            display_cols.append(col)

    print(pivot[display_cols].to_string(index=False, float_format='{:.2f}'.format))
    print("=" * 120)


if __name__ == '__main__':
    main()
