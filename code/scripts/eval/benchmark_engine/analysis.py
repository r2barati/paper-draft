"""
analysis.py — KPI computation, ablation metrics, sanity checks, regression detection.
"""

import json
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from .config import REFERENCE_FILE


# ---------------------------------------------------------------------------
# Ablation / Statistical Analysis (for Tier 3)
# ---------------------------------------------------------------------------

def compute_ablation_metrics(df):
    """
    Compute Value of Perfect Information (VPI), Value of Stochastic Solution (VSS),
    and Value of Partial Forecast (VPF) with paired t-test p-values.
    Returns a dict of ablation metrics for one scenario.
    """
    res = {}

    def _safe_mean(series):
        return series.mean() if len(series) > 0 else np.nan

    def _safe_std(series):
        return series.std() if len(series) > 0 else np.nan

    agents_present = df['agent'].unique()

    for agent in agents_present:
        agent_profits = df[df['agent'] == agent]['profit']
        res[f'{agent}_Profit'] = _safe_mean(agent_profits)
        res[f'{agent}_Profit_Std'] = _safe_std(agent_profits)
        res[f'{agent}_FillRate'] = _safe_mean(df[df['agent'] == agent]['fill_rate'])

    # VPI = Oracle - MSSP
    if 'Oracle' in agents_present and 'MSSP' in agents_present:
        res['VPI'] = res.get('Oracle_Profit', np.nan) - res.get('MSSP_Profit', np.nan)
        oracle_p = df[df['agent'] == 'Oracle']['profit'].values
        mssp_p = df[df['agent'] == 'MSSP']['profit'].values
        if len(oracle_p) > 1 and len(mssp_p) > 1 and len(oracle_p) == len(mssp_p):
            res['VPI_pval'] = ttest_rel(oracle_p, mssp_p)[1]
        else:
            res['VPI_pval'] = np.nan

    # VSS = MSSP - DLP
    if 'MSSP' in agents_present and 'DLP' in agents_present:
        res['VSS'] = res.get('MSSP_Profit', np.nan) - res.get('DLP_Profit', np.nan)
        mssp_p = df[df['agent'] == 'MSSP']['profit'].values
        dlp_p = df[df['agent'] == 'DLP']['profit'].values
        if len(mssp_p) > 1 and len(dlp_p) > 1 and len(mssp_p) == len(dlp_p):
            res['VSS_pval'] = ttest_rel(mssp_p, dlp_p)[1]
        else:
            res['VSS_pval'] = np.nan

    # VPF = Informed - Blind for each agent pair
    for base_name in ['MSSP', 'DLP', 'Heuristic']:
        blind_name = f'{base_name}_Blind'
        if base_name in agents_present and blind_name in agents_present:
            res[f'VPF_{base_name}'] = res.get(f'{base_name}_Profit', np.nan) - res.get(f'{blind_name}_Profit', np.nan)

    return res


# ---------------------------------------------------------------------------
# Sanity Checks (Tiers 1 & 2)
# ---------------------------------------------------------------------------

def check_sanity(results_df):
    """
    Run structural sanity checks on benchmark results.
    Returns (passed: bool, messages: list[str]).
    """
    messages = []
    passed = True

    if results_df.empty:
        return False, ['❌ No results collected — nothing to check.']

    # 1. No NaN profits
    nan_rows = results_df[results_df['profit'].isna()]
    if len(nan_rows) > 0:
        agents = nan_rows['agent'].unique().tolist()
        messages.append(f'❌ NaN profits detected for agents: {agents}')
        passed = False
    else:
        messages.append('✅ No NaN profits')

    # 2. Fill rates in [0, 1]
    if 'fill_rate' in results_df.columns:
        bad_fill = results_df[(results_df['fill_rate'] < 0) | (results_df['fill_rate'] > 1)]
        if len(bad_fill) > 0:
            messages.append(f'❌ Invalid fill rates detected ({len(bad_fill)} rows)')
            passed = False
        else:
            messages.append('✅ All fill rates in [0, 1]')

    # 3. Hierarchy: Oracle ≥ MSSP ≥ Dummy (on mean profit per scenario)
    agents_present = results_df['agent'].unique()
    mean_profits = results_df.groupby('agent')['profit'].mean()

    hierarchy_pairs = [
        ('Oracle', 'MSSP'),
        ('Oracle', 'DLP'),
        ('MSSP', 'DLP'),
        ('DLP', 'Dummy'),
        ('Heuristic', 'Dummy'),
    ]

    for better, worse in hierarchy_pairs:
        if better in agents_present and worse in agents_present:
            if mean_profits[better] < mean_profits[worse]:
                messages.append(f'⚠️  Hierarchy violation: {better} ({mean_profits[better]:.1f}) < {worse} ({mean_profits[worse]:.1f})')
                # This is a warning not a failure since it could happen on 1 seed
            else:
                messages.append(f'✅ {better} ({mean_profits[better]:.1f}) ≥ {worse} ({mean_profits[worse]:.1f})')

    # 4. No negative profits for Oracle
    if 'Oracle' in agents_present:
        oracle_negative = results_df[(results_df['agent'] == 'Oracle') & (results_df['profit'] < 0)]
        if len(oracle_negative) > 0:
            messages.append(f'❌ Oracle produced negative profits ({len(oracle_negative)} episodes)')
            passed = False
        else:
            messages.append('✅ Oracle profits all positive')

    return passed, messages


# ---------------------------------------------------------------------------
# Regression Detection (Tier 2+)
# ---------------------------------------------------------------------------

def check_regression(results_df, reference_path=None, threshold=0.15):
    """
    Compare current results against saved reference values.
    Returns (passed: bool, messages: list[str]).
    """
    ref_path = reference_path or REFERENCE_FILE
    if not os.path.exists(ref_path):
        return True, ['ℹ️  No reference file found — skipping regression check. Run with --save-reference to create one.']

    with open(ref_path, 'r') as f:
        reference = json.load(f)

    messages = []
    passed = True

    # Compare mean profit per agent
    mean_profits = results_df.groupby('agent')['profit'].mean()

    for agent, ref_profit in reference.get('agent_profits', {}).items():
        if agent not in mean_profits:
            messages.append(f'ℹ️  {agent}: not present in current run (skipped)')
            continue

        current = mean_profits[agent]
        if ref_profit == 0:
            continue

        delta_pct = (current - ref_profit) / abs(ref_profit)

        if delta_pct < -threshold:
            messages.append(f'🔴 {agent}: profit={current:.1f} (ref: {ref_profit:.1f}, Δ={delta_pct:+.1%}) ← REGRESSION')
            passed = False
        elif delta_pct > threshold:
            messages.append(f'🟡 {agent}: profit={current:.1f} (ref: {ref_profit:.1f}, Δ={delta_pct:+.1%}) ← IMPROVEMENT')
        else:
            messages.append(f'🟢 {agent}: profit={current:.1f} (ref: {ref_profit:.1f}, Δ={delta_pct:+.1%})')

    return passed, messages


def save_reference(results_df, reference_path=None):
    """Save current results as the regression reference."""
    ref_path = reference_path or REFERENCE_FILE
    os.makedirs(os.path.dirname(ref_path), exist_ok=True)

    mean_profits = results_df.groupby('agent')['profit'].mean().to_dict()
    mean_fill = results_df.groupby('agent')['fill_rate'].mean().to_dict()

    reference = {
        'agent_profits': {k: round(v, 2) for k, v in mean_profits.items()},
        'agent_fill_rates': {k: round(v, 4) for k, v in mean_fill.items()},
        'num_scenarios': len(results_df.groupby(['network', 'demand', 'goodwill', 'backlog'])),
        'num_seeds': len(results_df['seed'].unique()),
    }

    with open(ref_path, 'w') as f:
        json.dump(reference, f, indent=2)

    print(f'✅ Reference saved to {ref_path}')
