#!/usr/bin/env python3
"""
benchmark_dlp_adaptive_horizon.py — Adaptive-Horizon DLP Experiment.

Compares the original DLP (H=10) against an adaptive-horizon DLP where:
  H = min(max_cumulative_pipeline_lead_time + 10, num_periods - current_t)

This addresses the serial-network pipeline bottleneck by giving the DLP
enough foresight to pre-position inventory for demand shocks.

  Serial network: max_cumulative_LT = 8  → H_base = 18
  Base   network: max_cumulative_LT = 12 → H_base = 22

Output:
  data/results/dlp_adaptive_horizon_64.csv          (long-form)
  data/results/dlp_adaptive_vs_original_64.csv      (comparison)
"""

import os, sys, time
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.envs.builder import make_supply_chain_env
from src.agents.dlp_agent import RollingHorizonDLPAgent
from scripts.eval.benchmark_engine.runners import run_oracle, compute_kpi

# ---------------------------------------------------------------------------
# Scenario Matrix (same as original 64-scenario benchmark)
# ---------------------------------------------------------------------------
ALL_NETWORKS = ['base', 'serial']

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
NUM_PERIODS    = 30
DEFAULT_SEEDS  = list(range(100, 105))

RESULTS_DIR = 'data/results'

# ---------------------------------------------------------------------------
# Compute max cumulative pipeline lead time per network
# ---------------------------------------------------------------------------
def compute_max_pipeline_lt(network_name):
    """Compute the max cumulative lead time from raw material to retailer."""
    from src.envs.core.network_topology import SupplyChainNetwork
    import networkx as nx

    net = SupplyChainNetwork(scenario=network_name)

    # Find all paths from raw material sources to retail nodes
    max_lt = 0
    for rm in net.rawmat:
        for ret in net.retail:
            for path in nx.all_simple_paths(net.graph, rm, ret):
                # Reverse because edges go retailer→market direction in the graph
                # Actually, edges go upstream→downstream: (src, dst)
                path_lt = 0
                for i in range(len(path) - 1):
                    edge = (path[i], path[i+1])
                    if edge in net.graph.edges():
                        path_lt += net.graph.edges[edge].get('L', 0)
                    else:
                        # Try reverse
                        edge_rev = (path[i+1], path[i])
                        if edge_rev in net.graph.edges():
                            path_lt += net.graph.edges[edge_rev].get('L', 0)
                max_lt = max(max_lt, path_lt)

    # Also check by walking the reorder links (more reliable)
    # Build adjacency from reorder links
    reorder_lt = {}
    for (u, v) in net.reorder_links:
        reorder_lt[(u, v)] = net.graph.edges[(u, v)].get('L', 0)

    # BFS from raw materials through reorder links
    from collections import deque
    max_cumulative = 0
    for rm in net.rawmat:
        queue = deque([(rm, 0)])
        visited = set()
        while queue:
            node, cumlt = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            max_cumulative = max(max_cumulative, cumlt)
            for (u, v) in net.reorder_links:
                if u == node:
                    lt = net.graph.edges[(u, v)].get('L', 0)
                    queue.append((v, cumlt + lt))

    return max(max_lt, max_cumulative)


# ---------------------------------------------------------------------------
# Run DLP with specific planning horizon
# ---------------------------------------------------------------------------
def run_dlp_with_horizon(env_kwargs, seed, num_periods, agent_horizon, is_blind=False):
    """Run DLP agent with a specific planning horizon (not hardcoded to 10)."""
    env = make_supply_chain_env(agent_type='or', use_integer_actions=True, **env_kwargs)
    obs, _ = env.reset(seed=seed)

    # KEY DIFFERENCE: Use agent_horizon instead of hardcoded 10
    agent = RollingHorizonDLPAgent(
        env.unwrapped,
        planning_horizon=agent_horizon,
        is_blind=is_blind
    )

    episode_reward = 0
    for t in range(num_periods):
        # Optionally shrink horizon at end of episode
        remaining = num_periods - t
        if remaining < agent.H:
            agent.H = remaining

        obs, reward, terminated, truncated, _ = env.step(agent.get_action(t))
        episode_reward += reward
        if terminated or truncated:
            break

    return compute_kpi(env, episode_reward)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Adaptive Horizon DLP Experiment')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--skip-oracle', action='store_true',
                        help='Skip Oracle (use existing results from dlp_oracle_full_64.csv)')
    args = parser.parse_args()

    seeds = list(range(100, 100 + args.seeds))

    # Compute adaptive horizons per network
    horizon_map = {}
    for net_name in ALL_NETWORKS:
        max_lt = compute_max_pipeline_lt(net_name)
        adaptive_h = min(max_lt + 10, NUM_PERIODS)
        horizon_map[net_name] = {'max_lt': max_lt, 'H_original': 10, 'H_adaptive': adaptive_h}

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Adaptive Horizon DLP Experiment                           ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    for net_name, hinfo in horizon_map.items():
        print(f"║  {net_name:8s}: max_pipeline_LT={hinfo['max_lt']:2d}  "
              f"H_original={hinfo['H_original']:2d}  H_adaptive={hinfo['H_adaptive']:2d}")
    print(f"║  Seeds:       {seeds}")
    print(f"║  Scenarios:   64")
    print(f"║  Skip Oracle: {args.skip_oracle}")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Build scenario list
    scenarios = []
    for net in ALL_NETWORKS:
        for dlabel in ALL_DEMANDS:
            for gw in GOODWILL_FLAGS:
                for bl in BACKLOG_FLAGS:
                    scenarios.append((net, dlabel, gw, bl))

    # Agent configs: (display_name, is_blind, horizon_key)
    agents = [
        ('DLP_H10',       False, 'H_original'),
        ('DLP_H10_Blind', True,  'H_original'),
        ('DLP_Adaptive',       False, 'H_adaptive'),
        ('DLP_Adaptive_Blind', True,  'H_adaptive'),
    ]
    if not args.skip_oracle:
        agents.insert(0, ('Oracle', None, None))  # Oracle has no horizon

    all_results = []
    t_start = time.time()

    for idx, (net, dlabel, gw, bl) in enumerate(tqdm(scenarios, desc="Scenarios")):
        demand_config = dict(ALL_DEMANDS[dlabel])
        demand_config['use_goodwill'] = gw
        demand_config['base_mu'] = 20

        env_kwargs = {
            'scenario': net,
            'num_periods': NUM_PERIODS,
            'backlog': bl,
            'demand_config': demand_config,
        }

        scenario_tag = f"{net}|{dlabel}|GW={gw}|BL={bl}"
        tqdm.write(f"\n[{idx+1}/{len(scenarios)}] {scenario_tag}")

        hinfo = horizon_map[net]

        for agent_name, is_blind, h_key in agents:
            profits, fills, invs, unfulfilled, times = [], [], [], [], []

            for seed in seeds:
                t0 = time.time()
                try:
                    if agent_name == 'Oracle':
                        kpi = run_oracle(env_kwargs, seed, NUM_PERIODS)
                    else:
                        h = hinfo[h_key]
                        kpi = run_dlp_with_horizon(env_kwargs, seed, NUM_PERIODS, h, is_blind)
                    elapsed = time.time() - t0

                    profits.append(kpi['profit'])
                    fills.append(kpi['fill_rate'])
                    invs.append(kpi['avg_inv'])
                    unfulfilled.append(kpi['unfulfilled'])
                    times.append(elapsed)
                except Exception as e:
                    tqdm.write(f"  ⚠️  {agent_name} failed (seed={seed}): {e}")

            if len(profits) > 0:
                h_val = hinfo.get(h_key, '-') if h_key else '-'
                all_results.append({
                    'Network': net,
                    'Demand': dlabel,
                    'Goodwill': gw,
                    'Backlog': bl,
                    'Agent': agent_name,
                    'Horizon': h_val if isinstance(h_val, int) else 0,
                    'Profit_Mean': np.mean(profits),
                    'Profit_Std': np.std(profits),
                    'FillRate_Mean': np.mean(fills),
                    'AvgInv_Mean': np.mean(invs),
                    'Unfulfilled_Mean': np.mean(unfulfilled),
                    'Time_Mean_Sec': np.mean(times),
                    'Num_Seeds': len(profits),
                })
                tqdm.write(f"  {agent_name:20s} [H={h_val:>2}] profit={np.mean(profits):8.1f} ± "
                           f"{np.std(profits):6.1f}  fill={np.mean(fills):.3f}  t={np.mean(times):.2f}s")

    elapsed_total = time.time() - t_start

    # Save long-form
    df = pd.DataFrame(all_results)
    out_long = os.path.join(RESULTS_DIR, 'dlp_adaptive_horizon_64.csv')
    df.to_csv(out_long, index=False)
    print(f"\n✅ Results saved to {out_long}")
    print(f"   Total time: {elapsed_total/60:.1f} minutes")

    # ---------------------------------------------------------------------------
    # Generate comparison table: Original H=10 vs Adaptive
    # ---------------------------------------------------------------------------
    # Focus on DLP (informed) comparison
    dlp_h10  = df[df['Agent'] == 'DLP_H10'].set_index(['Network', 'Demand', 'Goodwill', 'Backlog'])
    dlp_adap = df[df['Agent'] == 'DLP_Adaptive'].set_index(['Network', 'Demand', 'Goodwill', 'Backlog'])

    if args.skip_oracle:
        # Load original Oracle results
        orig = pd.read_csv(os.path.join(RESULTS_DIR, 'dlp_oracle_full_64.csv'))
        oracle = orig[orig['Agent'] == 'Oracle'].set_index(['Network', 'Demand', 'Goodwill', 'Backlog'])
    else:
        oracle = df[df['Agent'] == 'Oracle'].set_index(['Network', 'Demand', 'Goodwill', 'Backlog'])

    comp_rows = []
    for idx_key in dlp_h10.index:
        row = {'Network': idx_key[0], 'Demand': idx_key[1],
               'Goodwill': idx_key[2], 'Backlog': idx_key[3]}

        if idx_key in oracle.index:
            row['Oracle_Profit'] = oracle.loc[idx_key, 'Profit_Mean']
        if idx_key in dlp_h10.index:
            row['DLP_H10_Profit'] = dlp_h10.loc[idx_key, 'Profit_Mean']
            row['DLP_H10_Fill'] = dlp_h10.loc[idx_key, 'FillRate_Mean']
            row['DLP_H10_Time'] = dlp_h10.loc[idx_key, 'Time_Mean_Sec']
        if idx_key in dlp_adap.index:
            row['DLP_Adaptive_Profit'] = dlp_adap.loc[idx_key, 'Profit_Mean']
            row['DLP_Adaptive_Fill'] = dlp_adap.loc[idx_key, 'FillRate_Mean']
            row['DLP_Adaptive_Time'] = dlp_adap.loc[idx_key, 'Time_Mean_Sec']

        # Compute improvement
        if 'DLP_H10_Profit' in row and 'DLP_Adaptive_Profit' in row:
            row['Improvement'] = row['DLP_Adaptive_Profit'] - row['DLP_H10_Profit']
            if row['DLP_H10_Profit'] != 0:
                row['Improvement_%'] = (row['Improvement'] / abs(row['DLP_H10_Profit'])) * 100
            else:
                row['Improvement_%'] = float('inf') if row['Improvement'] > 0 else 0

        if 'Oracle_Profit' in row and 'DLP_H10_Profit' in row:
            row['Gap_H10_%'] = ((row['Oracle_Profit'] - row['DLP_H10_Profit']) / abs(row['Oracle_Profit'])) * 100
        if 'Oracle_Profit' in row and 'DLP_Adaptive_Profit' in row:
            row['Gap_Adaptive_%'] = ((row['Oracle_Profit'] - row['DLP_Adaptive_Profit']) / abs(row['Oracle_Profit'])) * 100

        comp_rows.append(row)

    comp_df = pd.DataFrame(comp_rows)
    out_comp = os.path.join(RESULTS_DIR, 'dlp_adaptive_vs_original_64.csv')
    comp_df.to_csv(out_comp, index=False)
    print(f"✅ Comparison saved to {out_comp}")

    # Summary stats
    print("\n" + "=" * 100)
    print("  Adaptive Horizon DLP — Summary of Improvements")
    print("=" * 100)

    for net in ALL_NETWORKS:
        net_df = comp_df[comp_df['Network'] == net]
        print(f"\n--- {net.upper()} Network (H: 10 → {horizon_map[net]['H_adaptive']}) ---")
        print(f"  Mean improvement: ${net_df['Improvement'].mean():.1f}")
        print(f"  Max  improvement: ${net_df['Improvement'].max():.1f}")
        improved = (net_df['Improvement'] > 1).sum()
        print(f"  Scenarios improved: {improved}/{len(net_df)}")
        if 'Gap_H10_%' in net_df.columns:
            print(f"  Mean gap H=10:      {net_df['Gap_H10_%'].mean():.1f}%")
            print(f"  Mean gap adaptive:  {net_df['Gap_Adaptive_%'].mean():.1f}%")

    # Print worst scenarios improvement
    print("\n--- Biggest Improvements (top 10) ---")
    top = comp_df.nlargest(10, 'Improvement')
    for _, r in top.iterrows():
        print(f"  {r['Network']:7s}|{r['Demand']:25s}|GW={r['Goodwill']}|BL={r['Backlog']}: "
              f"H10={r['DLP_H10_Profit']:8.1f} → Adaptive={r['DLP_Adaptive_Profit']:8.1f}  "
              f"Δ={r['Improvement']:+.1f}")

    print("=" * 100)


if __name__ == '__main__':
    main()
