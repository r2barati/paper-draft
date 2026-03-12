#!/usr/bin/env python3
"""
benchmark_combined_effects.py — Benchmark MSSP (+ all OR baselines) on combined demand effects.

The DemandEngine supports composable effects (trend+seasonal, trend+shock, etc.)
that were never benchmarked. This script evaluates all 4 new combined-effect
demand configurations across the full scenario matrix.

Combined demand configs tested:
  - trend_seasonal:       effects=['trend', 'seasonal']
  - trend_shock:          effects=['trend', 'shock']
  - seasonal_shock:       effects=['seasonal', 'shock']
  - trend_seasonal_shock: effects=['trend', 'seasonal', 'shock']

Each is tested on:
  - 2 networks (base, serial)
  - 2 goodwill flags (False, True)
  - 2 backlog flags (True, False)
  - 5 seeds (42–46)

With agents: Oracle, MSSP, MSSP_Blind, DLP, DLP_Blind, Heuristic, Heuristic_Blind, Dummy

Usage:
    python scripts/eval/benchmark_combined_effects.py
    python scripts/eval/benchmark_combined_effects.py --networks base --seeds 3
    python scripts/eval/benchmark_combined_effects.py --networks serial --demand trend_seasonal
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import itertools
import time
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from src.envs.builder import make_supply_chain_env
from src.envs.core.environment import CoreEnv
from src.agents.oracle import StandaloneOracleOptimizer
from src.agents.dlp_agent import RollingHorizonDLPAgent
from src.agents.mssp_agent import RollingHorizonMSSPAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.baselines import EndogenousOracle, OptimisticEndogenousOracle


# ---------------------------------------------------------------------------
# Combined Demand Configurations
# ---------------------------------------------------------------------------
COMBINED_DEMAND_CONFIGS = {
    'trend_seasonal': {
        'type': 'trend_seasonal',
        'effects': ['trend', 'seasonal'],
        'base_mu': 20,
    },
    'trend_shock': {
        'type': 'trend_shock',
        'effects': ['trend', 'shock'],
        'base_mu': 20,
    },
    'seasonal_shock': {
        'type': 'seasonal_shock',
        'effects': ['seasonal', 'shock'],
        'base_mu': 20,
    },
    'trend_seasonal_shock': {
        'type': 'trend_seasonal_shock',
        'effects': ['trend', 'seasonal', 'shock'],
        'base_mu': 20,
    },
}

PLANNING_HORIZON = 30
OUTPUT_CSV = 'data/results/benchmark_results_combined_effects.csv'


# ---------------------------------------------------------------------------
# KPI extraction
# ---------------------------------------------------------------------------
def _kpi(env, episode_reward):
    total_demand = np.sum(env.D)
    total_sales = 0.0
    for i, (j, k) in enumerate(env.network.retail_links):
        net_idx = env.network.network_map[(j, k)]
        total_sales += np.sum(env.S[:, net_idx])

    fill_rate = total_sales / total_demand if total_demand > 0 else 1.0

    main_indices = [env.network.node_map[n] for n in env.network.main_nodes]
    avg_inv = np.mean(np.sum(env.X[:env.num_periods, main_indices], axis=1))

    return {
        'profit': np.sum(env.P),
        'fill_rate': fill_rate,
        'avg_inv': avg_inv,
        'unfulfilled': np.sum(env.U),
        'avg_backlog': np.mean(np.sum(env.U, axis=1)),
        'final_sentiment': env.demand_engine.sentiment,
    }


# ---------------------------------------------------------------------------
# Agent Runners (following same pattern as benchmark_iterative.py)
# ---------------------------------------------------------------------------
def _run_oracle(env_kwargs, seed, planning_horizon):
    """Run the Oracle agent. Uses EndogenousOracle for goodwill scenarios."""
    use_gw = env_kwargs.get('demand_config', {}).get('use_goodwill', False)

    if use_gw:
        oracle = OptimisticEndogenousOracle(env_kwargs, planning_horizon)
        actions = oracle.solve(seed)
        if actions is None:
            return None

        env = CoreEnv(**env_kwargs)
        obs, _ = env.reset(seed=seed)
        episode_reward = 0
        for t in range(planning_horizon):
            obs, reward, terminated, truncated, _ = env.step(actions[t])
            episode_reward += reward
            if terminated or truncated:
                break
        return _kpi(env, episode_reward)
    else:
        # Non-goodwill: generate demand trace, then optimize with static Oracle
        env = CoreEnv(**env_kwargs)
        obs, _ = env.reset(seed=seed)
        # Run with zero actions to extract actual demand trace
        for t in range(planning_horizon):
            env.step(np.zeros(env.action_space.shape))

        demand_trace = {}
        for i, edge in enumerate(env.network.retail_links):
            demand_trace[edge] = env.D[:, i].copy()

        # Rebuild env with fixed demand
        fixed_cfg = env_kwargs.copy()
        fixed_cfg['user_D'] = demand_trace
        env2 = CoreEnv(**fixed_cfg)
        env2.reset(seed=seed)

        oracle = StandaloneOracleOptimizer(
            env2, known_demand_scenario=demand_trace,
            planning_horizon=planning_horizon, is_continuous=True
        )
        actions = oracle.solve_full_horizon()
        if actions is None:
            return None

        # Replay with oracle actions
        env3 = CoreEnv(**env_kwargs)
        env3.reset(seed=seed)
        episode_reward = 0
        for t in range(planning_horizon):
            obs, reward, terminated, truncated, _ = env3.step(actions[t])
            episode_reward += reward
            if terminated or truncated:
                break
        return _kpi(env3, episode_reward)


def _run_mssp(env_kwargs, seed, planning_horizon, is_blind=False):
    env = make_supply_chain_env(agent_type='or', use_integer_actions=True, **env_kwargs)
    obs, _ = env.reset(seed=seed)
    agent = RollingHorizonMSSPAgent(env.unwrapped, planning_horizon=10, branching_depth=3, is_blind=is_blind)
    episode_reward = 0
    for t in range(planning_horizon):
        obs, reward, terminated, truncated, _ = env.step(agent.get_action(t))
        episode_reward += reward
        if terminated or truncated:
            break
    return _kpi(env.unwrapped, episode_reward)


def _run_dlp(env_kwargs, seed, planning_horizon, is_blind=False):
    env = make_supply_chain_env(agent_type='or', use_integer_actions=True, **env_kwargs)
    obs, _ = env.reset(seed=seed)
    agent = RollingHorizonDLPAgent(env.unwrapped, planning_horizon=10, is_blind=is_blind)
    episode_reward = 0
    for t in range(planning_horizon):
        obs, reward, terminated, truncated, _ = env.step(agent.get_action(t))
        episode_reward += reward
        if terminated or truncated:
            break
    return _kpi(env.unwrapped, episode_reward)


def _run_heuristic(env_kwargs, seed, planning_horizon, is_blind=False):
    env = make_supply_chain_env(agent_type='or', use_integer_actions=True, **env_kwargs)
    obs, _ = env.reset(seed=seed)
    agent = HeuristicAgent(env.unwrapped, is_blind=is_blind)
    episode_reward = 0
    for t in range(planning_horizon):
        obs, reward, terminated, truncated, _ = env.step(agent.get_action(obs, t))
        episode_reward += reward
        if terminated or truncated:
            break
    return _kpi(env.unwrapped, episode_reward)


def _run_dummy(env_kwargs, seed, planning_horizon):
    env = make_supply_chain_env(agent_type='or', use_integer_actions=True, **env_kwargs)
    obs, _ = env.reset(seed=seed)
    episode_reward = 0
    for t in range(planning_horizon):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        episode_reward += reward
        if terminated or truncated:
            break
    return _kpi(env.unwrapped, episode_reward)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_acc = lambda metrics, res: [metrics[k].append(res[k]) for k in metrics]
_mean = lambda lst: np.mean(lst) if lst else np.nan
_std = lambda lst: np.std(lst) if lst else np.nan


# ---------------------------------------------------------------------------
# Main evaluation block
# ---------------------------------------------------------------------------
def evaluate_scenario(scenario, max_episodes, planning_horizon):
    """Evaluate all OR baseline agents on a single combined-effect scenario."""
    demand_cfg = COMBINED_DEMAND_CONFIGS[scenario['demand_type']].copy()
    demand_cfg['use_goodwill'] = scenario['use_goodwill']

    env_kwargs = {
        'scenario': scenario['network'],
        'backlog': scenario['backlog'],
        'demand_config': demand_cfg,
        'num_periods': planning_horizon,
    }

    metrics = {agent: {'profit': [], 'fill_rate': [], 'avg_inv': [], 'unfulfilled': []}
               for agent in ['Oracle', 'MSSP', 'MSSP_Blind', 'DLP', 'DLP_Blind',
                             'Dummy', 'Heuristic', 'Heuristic_Blind']}

    t_agent = {agent: [] for agent in metrics.keys()}
    scenario_start = time.perf_counter()

    for episode in range(max_episodes):
        seed = 42 + episode

        def track(a_name, func, *args, **kwargs):
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            if result is not None:
                _acc(metrics[a_name], result)
            t_agent[a_name].append(time.perf_counter() - t0)

        track('Oracle', _run_oracle, env_kwargs, seed, planning_horizon)
        track('MSSP', _run_mssp, env_kwargs, seed, planning_horizon, is_blind=False)
        track('MSSP_Blind', _run_mssp, env_kwargs, seed, planning_horizon, is_blind=True)
        track('DLP', _run_dlp, env_kwargs, seed, planning_horizon, is_blind=False)
        track('DLP_Blind', _run_dlp, env_kwargs, seed, planning_horizon, is_blind=True)
        track('Dummy', _run_dummy, env_kwargs, seed, planning_horizon)
        track('Heuristic', _run_heuristic, env_kwargs, seed, planning_horizon, is_blind=False)
        track('Heuristic_Blind', _run_heuristic, env_kwargs, seed, planning_horizon, is_blind=True)

    elapsed = time.perf_counter() - scenario_start

    # Compile results (same format as benchmark_iterative.py)
    res = {
        'Network': scenario['network'],
        'Demand': scenario['demand_type'],
        'Goodwill': scenario['use_goodwill'],
        'Backlog': scenario['backlog'],
    }

    for agent, m in metrics.items():
        res[f'{agent}_Profit'] = _mean(m['profit'])
        res[f'{agent}_FillRate'] = _mean(m['fill_rate'])
        res[f'{agent}_AvgInv'] = _mean(m['avg_inv'])
        res[f'{agent}_Unfulfilled'] = _mean(m['unfulfilled'])

    res['Oracle_Profit_Std'] = _std(metrics['Oracle']['profit'])
    res['MSSP_Profit_Std'] = _std(metrics['MSSP']['profit'])
    res['DLP_Profit_Std'] = _std(metrics['DLP']['profit'])

    res['VPI'] = _mean(metrics['Oracle']['profit']) - _mean(metrics['MSSP']['profit'])
    res['VSS'] = _mean(metrics['MSSP']['profit']) - _mean(metrics['DLP']['profit'])
    res['VPF_MSSP'] = _mean(metrics['MSSP']['profit']) - _mean(metrics['MSSP_Blind']['profit'])
    res['VPF_DLP'] = _mean(metrics['DLP']['profit']) - _mean(metrics['DLP_Blind']['profit'])
    res['VPF_Heuristic'] = _mean(metrics['Heuristic']['profit']) - _mean(metrics['Heuristic_Blind']['profit'])

    if len(metrics['Oracle']['profit']) > 1 and len(metrics['MSSP']['profit']) > 1:
        res['VPI_pval'] = ttest_rel(metrics['Oracle']['profit'], metrics['MSSP']['profit'])[1]
    else:
        res['VPI_pval'] = np.nan

    if len(metrics['MSSP']['profit']) > 1 and len(metrics['DLP']['profit']) > 1:
        res['VSS_pval'] = ttest_rel(metrics['MSSP']['profit'], metrics['DLP']['profit'])[1]
    else:
        res['VSS_pval'] = np.nan

    for agent, t in t_agent.items():
        res[f'Time_{agent}'] = np.mean(t) if t else np.nan
    res['Time_Sec_Total'] = elapsed

    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Benchmark combined demand effects')
    parser.add_argument('--networks', nargs='+', default=['base', 'serial'],
                        help='Networks to test (default: base serial)')
    parser.add_argument('--demand', nargs='+', default=list(COMBINED_DEMAND_CONFIGS.keys()),
                        help='Combined demand configs to test')
    parser.add_argument('--seeds', type=int, default=5,
                        help='Number of seeds per scenario (default: 5)')
    parser.add_argument('--output', type=str, default=OUTPUT_CSV,
                        help='Output CSV path')
    args = parser.parse_args()

    networks = args.networks
    demand_types = args.demand
    goodwill_flags = [False, True]
    backlog_flags = [True, False]

    scenarios = []
    for net, dem, gw, bl in itertools.product(networks, demand_types, goodwill_flags, backlog_flags):
        scenarios.append({
            'network': net,
            'demand_type': dem,
            'use_goodwill': gw,
            'backlog': bl,
        })

    print("=" * 70)
    print(f"  Combined-Effect Benchmark")
    print(f"  {len(scenarios)} scenarios × {args.seeds} seeds × 8 agents")
    print(f"  Demand configs: {demand_types}")
    print(f"  Networks: {networks}")
    print("=" * 70)

    all_results = []

    for i, scen in enumerate(scenarios):
        label = f"{scen['network']:6s} | {scen['demand_type']:22s} | GW={scen['use_goodwill']} | BL={scen['backlog']}"
        print(f"\n  [{i+1}/{len(scenarios)}] {label} ...", flush=True)

        try:
            res = evaluate_scenario(scen, args.seeds, PLANNING_HORIZON)
            all_results.append(res)
            print(f"    Oracle={res['Oracle_Profit']:.1f}  MSSP={res['MSSP_Profit']:.1f}  "
                  f"DLP={res['DLP_Profit']:.1f}  Heuristic={res['Heuristic_Profit']:.1f}  "
                  f"VPI={res['VPI']:.1f}  VSS={res['VSS']:.1f}  VPF_MSSP={res['VPF_MSSP']:.1f}")
        except Exception as e:
            print(f"    ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    df = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n{'=' * 70}")
    print(f"  Results saved to {args.output}")
    print(f"  Total scenarios completed: {len(all_results)}/{len(scenarios)}")
    print(f"{'=' * 70}")

    # Quick summary
    if not df.empty:
        print("\n--- Summary Table ---")
        summary_cols = ['Network', 'Demand', 'Goodwill', 'Backlog',
                        'Oracle_Profit', 'MSSP_Profit', 'DLP_Profit', 'Heuristic_Profit',
                        'VPI', 'VSS', 'VPF_MSSP']
        available = [c for c in summary_cols if c in df.columns]
        print(df[available].to_string(index=False))


if __name__ == '__main__':
    main()
