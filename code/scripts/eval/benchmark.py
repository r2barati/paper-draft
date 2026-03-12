#!/usr/bin/env python3
"""
benchmark.py — Consolidated Three-Tier Progressive Benchmark

Usage:
    python scripts/eval/benchmark.py --tier 1           # Smoke test (~1-2 min)
    python scripts/eval/benchmark.py --tier 2           # Sanity check (~10-15 min)
    python scripts/eval/benchmark.py --tier 3 --seeds 30  # Comprehensive

    python scripts/eval/benchmark.py --tier 2 --agents or   # OR baselines only
    python scripts/eval/benchmark.py --tier 2 --agents rl   # RL models only

    python scripts/eval/benchmark.py --save-reference      # Save current as reference
    python scripts/eval/benchmark.py --tier 3 --no-cache   # Force re-run
"""

import sys
import os

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Thread safety for macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import itertools
import time
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from benchmark_engine.config import (
    TIER_CONFIG, PLANNING_HORIZON, AGENT_ORDER,
    RESULTS_DIR, LOG_DIR, CACHE_DIR, REFERENCE_FILE,
    DEMAND_TYPES, NETWORKS, GOODWILL_FLAGS, BACKLOG_FLAGS,
)
from benchmark_engine.runners import AGENT_REGISTRY
from benchmark_engine.analysis import (
    check_sanity, check_regression, save_reference, compute_ablation_metrics,
)


# ---------------------------------------------------------------------------
# Scenario Generation
# ---------------------------------------------------------------------------

def generate_scenarios(tier, seeds_override=None, network_override=None, demand_override=None):
    """Build list of scenario dicts for a given tier."""
    cfg = TIER_CONFIG[tier]

    networks = [network_override] if network_override else cfg['networks']
    demands = [demand_override] if demand_override else cfg['demand_types']
    goodwills = cfg['goodwill']
    backlogs = cfg['backlog']

    if seeds_override is not None:
        seeds = list(range(100, 100 + seeds_override))
    elif cfg['seeds'] is not None:
        seeds = cfg['seeds']
    else:
        seeds = list(range(100, 105))  # default for tier 3

    scenarios = []
    for net, dem, gw, bl in itertools.product(networks, demands, goodwills, backlogs):
        for seed in seeds:
            scenarios.append({
                'network': net,
                'demand_type': dem,
                'use_goodwill': gw,
                'backlog': bl,
                'seed': seed,
            })

    return scenarios, seeds


def scenario_key(s):
    """Unique key for a scenario (without seed)."""
    return (s['network'], s['demand_type'], s['use_goodwill'], s['backlog'])


# ---------------------------------------------------------------------------
# Cache Management
# ---------------------------------------------------------------------------

def get_cache_path(tier):
    return os.path.join(CACHE_DIR, f'benchmark_tier{tier}_cache.csv')


def load_cached_results(tier):
    """Load already-completed results from cache CSV."""
    cache_path = get_cache_path(tier)
    if not os.path.exists(cache_path):
        return pd.DataFrame()
    try:
        return pd.read_csv(cache_path)
    except Exception:
        return pd.DataFrame()


def is_completed(cached_df, scenario, agent_name):
    """Check if a specific (scenario, agent, seed) combo is already cached."""
    if cached_df.empty:
        return False
    mask = (
        (cached_df['network'] == scenario['network']) &
        (cached_df['demand'] == scenario['demand_type']) &
        (cached_df['goodwill'] == scenario['use_goodwill']) &
        (cached_df['backlog'] == scenario['backlog']) &
        (cached_df['seed'] == scenario['seed']) &
        (cached_df['agent'] == agent_name)
    )
    return mask.any()


def append_to_cache(tier, row_dict):
    """Append one result row to the tier's cache CSV."""
    cache_path = get_cache_path(tier)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df = pd.DataFrame([row_dict])
    df.to_csv(cache_path, mode='a', header=not os.path.exists(cache_path), index=False)


# ---------------------------------------------------------------------------
# Core Benchmark Runner
# ---------------------------------------------------------------------------

def run_tier(tier, agent_filter=None, use_cache=True, seeds_override=None,
             network_override=None, demand_override=None):
    """
    Execute a single tier of the benchmark.
    Returns (results_df, passed_sanity: bool).
    """
    cfg = TIER_CONFIG[tier]
    scenarios, seeds = generate_scenarios(
        tier, seeds_override=seeds_override,
        network_override=network_override, demand_override=demand_override,
    )

    # Determine which agents to run
    agents_to_run = []
    for agent_name in AGENT_ORDER:
        if agent_name not in AGENT_REGISTRY:
            continue
        reg = AGENT_REGISTRY[agent_name]
        if agent_filter and reg['category'] != agent_filter:
            continue
        agents_to_run.append(agent_name)

    total_scenarios = len(set(scenario_key(s) for s in scenarios))
    total_seeds = len(seeds)

    print(f"\n{'=' * 70}")
    print(f"  TIER {tier}: {cfg['description']}")
    print(f"  {total_scenarios} scenarios × {total_seeds} seed(s) × {len(agents_to_run)} agents")
    print(f"{'=' * 70}\n")

    # Load cache
    cached_df = load_cached_results(tier) if use_cache else pd.DataFrame()
    all_results = []

    # If we have cached results, include them
    if not cached_df.empty:
        all_results.extend(cached_df.to_dict('records'))

    # Group scenarios by (network, demand, goodwill, backlog) for progress display
    unique_scenarios = list(set(scenario_key(s) for s in scenarios))
    unique_scenarios.sort()

    for agent_name in agents_to_run:
        reg = AGENT_REGISTRY[agent_name]
        runner = reg['runner']
        agent_kwargs = reg.get('kwargs', {})

        print(f"  Agent: {agent_name} ({reg['category'].upper()})")

        skipped = 0
        failed = 0
        succeeded = 0
        agent_start = time.perf_counter()

        for scenario in tqdm(scenarios, desc=f"    {agent_name}", leave=True):
            # Skip if cached
            if use_cache and is_completed(cached_df, scenario, agent_name):
                skipped += 1
                continue

            env_kwargs = {
                'scenario': scenario['network'],
                'backlog': scenario['backlog'],
                'num_periods': PLANNING_HORIZON,
                'demand_config': {
                    'type': scenario['demand_type'],
                    'use_goodwill': scenario['use_goodwill'],
                    'base_mu': 20,
                },
            }

            try:
                result = runner(env_kwargs, scenario['seed'], PLANNING_HORIZON, **agent_kwargs)
            except Exception as e:
                print(f"\n    ❌ {agent_name} FAILED on {scenario_key(scenario)} seed={scenario['seed']}: {e}")
                failed += 1
                # For Tier 1: abort immediately on OR baselines (they should always work).
                # RL failures are logged as warnings and we continue.
                if tier == 1 and reg['category'] == 'or':
                    return pd.DataFrame(all_results), False
                continue

            if result is None:
                skipped += 1
                continue

            row = {
                'agent': agent_name,
                'network': scenario['network'],
                'demand': scenario['demand_type'],
                'goodwill': scenario['use_goodwill'],
                'backlog': scenario['backlog'],
                'seed': scenario['seed'],
                'profit': result['profit'],
                'fill_rate': result['fill_rate'],
                'avg_inv': result['avg_inv'],
                'unfulfilled': result['unfulfilled'],
                'avg_backlog': result['avg_backlog'],
                'final_sentiment': result['final_sentiment'],
            }

            all_results.append(row)
            if use_cache and tier >= 2:
                append_to_cache(tier, row)
            succeeded += 1

        elapsed = time.perf_counter() - agent_start
        print(f"    → {succeeded} new, {skipped} cached/skipped, {failed} failed ({elapsed:.1f}s)\n")

    results_df = pd.DataFrame(all_results)

    if results_df.empty:
        print("  ⚠️ No results collected.")
        return results_df, False

    # Run sanity checks
    print(f"\n{'─' * 50}")
    print(f"  Sanity Checks (Tier {tier})")
    print(f"{'─' * 50}")
    passed, messages = check_sanity(results_df)
    for msg in messages:
        print(f"  {msg}")

    # Run regression check (Tier 2+)
    if tier >= 2:
        print(f"\n{'─' * 50}")
        print(f"  Regression Check")
        print(f"{'─' * 50}")
        reg_passed, reg_messages = check_regression(results_df)
        for msg in reg_messages:
            print(f"  {msg}")
        passed = passed and reg_passed

    return results_df, passed


# ---------------------------------------------------------------------------
# Summary & Output
# ---------------------------------------------------------------------------

def print_summary(results_df, tier):
    """Print a human-readable summary table."""
    if results_df.empty:
        return

    print(f"\n{'=' * 70}")
    print(f"  RESULTS SUMMARY — Tier {tier}")
    print(f"{'=' * 70}\n")

    # Mean profit by agent across all scenarios
    summary = results_df.groupby('agent')['profit'].agg(['mean', 'std', 'count'])
    summary.columns = ['Mean Profit', 'Std', 'Episodes']
    summary = summary.sort_values('Mean Profit', ascending=False)
    print(summary.round(2).to_string())

    # Detailed breakdown by scenario
    if tier >= 2:
        print(f"\n{'─' * 50}")
        print("  Profit by Scenario (mean)")
        print(f"{'─' * 50}\n")
        pivot = results_df.groupby(
            ['network', 'demand', 'goodwill', 'agent']
        )['profit'].mean().unstack('agent')
        print(pivot.round(1).to_string())


def save_comprehensive_csv(results_df, tier):
    """Save results to the standard output CSV."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if tier == 3:
        out_path = os.path.join(RESULTS_DIR, 'benchmark_results_comprehensive.csv')
    else:
        out_path = os.path.join(RESULTS_DIR, f'benchmark_results_tier{tier}.csv')

    results_df.to_csv(out_path, index=False)
    print(f"\n  📄 Results saved to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Consolidated Three-Tier Progressive Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --tier 1                  Smoke test (~1-2 min)
  %(prog)s --tier 2                  Sanity check (~10-15 min)
  %(prog)s --tier 3 --seeds 30       Comprehensive with 30 seeds
  %(prog)s --tier 2 --agents or      OR baselines only
  %(prog)s --tier 2 --agents rl      RL models only
  %(prog)s --save-reference          Save current Tier 3 as reference
  %(prog)s --tier 3 --no-cache       Force re-run (ignore cache)
        """,
    )
    parser.add_argument('--tier', type=int, choices=[1, 2, 3], default=2,
                        help='Benchmark tier (1=smoke, 2=sanity, 3=comprehensive)')
    parser.add_argument('--agents', choices=['or', 'rl', 'all'], default='all',
                        help='Agent category filter')
    parser.add_argument('--seeds', type=int, default=None,
                        help='Override number of seeds (Tier 3 default: 5)')
    parser.add_argument('--network', choices=['base', 'serial'], default=None,
                        help='Run only on a specific network')
    parser.add_argument('--demand', default=None,
                        help='Run only on a specific demand type')
    parser.add_argument('--no-cache', action='store_true',
                        help='Ignore cached results, force re-run')
    parser.add_argument('--save-reference', action='store_true',
                        help='Save current Tier 2 results as regression reference')

    args = parser.parse_args()
    agent_filter = None if args.agents == 'all' else args.agents
    use_cache = not args.no_cache

    overall_start = time.perf_counter()

    # --- Save Reference Mode ---
    if args.save_reference:
        # Try loading the latest Tier 3 cache, fallback to Tier 2
        for t in [3, 2]:
            cache_path = get_cache_path(t)
            if os.path.exists(cache_path):
                df = pd.read_csv(cache_path)
                save_reference(df)
                return
        print("❌ No cached results found. Run --tier 2 or --tier 3 first.")
        return

    # --- Progressive Tier Execution ---
    target_tier = args.tier

    # Tier 2+ requires Tier 1 to pass first
    if target_tier >= 2:
        print("╔══════════════════════════════════════════════════════╗")
        print("║  Running Tier 1 (Smoke Test) before Tier", target_tier, "          ║")
        print("╚══════════════════════════════════════════════════════╝")
        _, tier1_passed = run_tier(
            1, agent_filter=agent_filter, use_cache=False,  # always re-run smoke test
        )
        if not tier1_passed:
            print("\n❌ TIER 1 SMOKE TEST FAILED — aborting before Tier", target_tier)
            print("   Fix the issues above before running a larger benchmark.")
            sys.exit(1)
        print("\n✅ Tier 1 passed — proceeding to Tier", target_tier)

    # Tier 3 requires Tier 2 to pass
    if target_tier >= 3:
        print("\n╔══════════════════════════════════════════════════════╗")
        print("║  Running Tier 2 (Sanity Check) before Tier 3        ║")
        print("╚══════════════════════════════════════════════════════╝")
        _, tier2_passed = run_tier(
            2, agent_filter=agent_filter, use_cache=use_cache,
        )
        if not tier2_passed:
            print("\n❌ TIER 2 SANITY CHECK FAILED — aborting before Tier 3")
            print("   Investigate the warnings above before running comprehensive benchmark.")
            sys.exit(1)
        print("\n✅ Tier 2 passed — proceeding to Tier 3")

    # Run target tier
    results_df, passed = run_tier(
        target_tier,
        agent_filter=agent_filter,
        use_cache=use_cache,
        seeds_override=args.seeds,
        network_override=args.network,
        demand_override=args.demand,
    )

    # Print summary
    print_summary(results_df, target_tier)

    # Save to CSV
    if not results_df.empty:
        save_comprehensive_csv(results_df, target_tier)

    # Compute ablation metrics for Tier 3
    if target_tier == 3 and not results_df.empty:
        print(f"\n{'─' * 50}")
        print("  Ablation Study")
        print(f"{'─' * 50}\n")
        ablation_rows = []
        for keys, group in results_df.groupby(['network', 'demand', 'goodwill', 'backlog']):
            metrics = compute_ablation_metrics(group)
            metrics['Network'], metrics['Demand'], metrics['Goodwill'], metrics['Backlog'] = keys
            ablation_rows.append(metrics)

        if ablation_rows:
            ablation_df = pd.DataFrame(ablation_rows)
            ablation_cols = ['Network', 'Demand', 'Goodwill', 'Backlog']
            metric_cols = [c for c in ablation_df.columns if c not in ablation_cols]
            display_cols = ablation_cols + sorted([c for c in metric_cols if 'VPI' in c or 'VSS' in c or 'VPF' in c])
            available = [c for c in display_cols if c in ablation_df.columns]
            if available:
                print(ablation_df[available].round(2).to_string(index=False))

    # Final status
    elapsed = time.perf_counter() - overall_start
    print(f"\n{'=' * 70}")
    status = "✅ ALL PASSED" if passed else "⚠️  COMPLETED WITH WARNINGS"
    print(f"  {status} — Total time: {elapsed:.1f}s")
    print(f"{'=' * 70}\n")

    if not passed:
        sys.exit(1)


if __name__ == '__main__':
    main()
