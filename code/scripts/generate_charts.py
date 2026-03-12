"""
generate_charts.py — Publication-quality basic benchmark charts.

Generates 6 core charts from the comprehensive benchmark CSV:
  01  Pareto Frontier (Cost–Service Trade-off)
  02  Algorithm Generalizability across Demand Profiles
  03  Relative Optimality Gap Distribution (Violin + Strip)
  04  Value of Information (VPI / VSS)
  05  Information Asymmetry (Aware vs Blind)
  06  Overall Average Profit
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from chart_style import (
    apply_style, AGENT_PALETTE, AGENT_ORDER, DEMAND_ORDER,
    BLIND_LABELS, save_fig, get_palette, get_agent_order,
    add_bar_labels, FIG_DOUBLE, FIG_SINGLE,
)

apply_style()

# ─────────────────────────────────────────────────────────────────────────────
#  Data loading & processing
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path):
    print(f"Loading data from {csv_path}...")
    try:
        return pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        return None


def process_data(df):
    """
    Convert wide CSV into a long-form DataFrame of (scenario, agent, metrics).
    Excludes Dummy (non-competitive) and DLP (redundant with MSSP).
    Filters out catastrophically-failed agents (e.g. RLV4 on serial topology
    where it produces -30k profits due to being trained on base only).
    """
    valid_agents = ['Oracle', 'MSSP', 'RLV4', 'Residual', 'RLGNN']
    valid_blind  = ['MSSP_Blind', 'DLP_Blind', 'Heuristic_Blind']

    rows = []
    for _, row in df.iterrows():
        network  = row['Network']
        demand   = row['Demand']
        goodwill = row['Goodwill']
        backlog  = row['Backlog']

        for agent in valid_agents:
            profit_col = f"{agent}_Profit"
            if profit_col in row and not pd.isna(row[profit_col]):
                profit = row[profit_col]
                # Skip catastrophic failures (trained on wrong topology)
                if profit < -5000:
                    continue
                rows.append({
                    'Network': network, 'Demand': demand,
                    'Goodwill': goodwill, 'Backlog': backlog,
                    'Agent': agent, 'Blind': False,
                    'Profit': profit,
                    'FillRate': row.get(f"{agent}_FillRate", np.nan),
                    'AvgInv': row.get(f"{agent}_AvgInv", np.nan),
                    'Unfulfilled': row.get(f"{agent}_Unfulfilled", np.nan),
                })

        for b_agent in valid_blind:
            b_profit_col = f"{b_agent}_Profit"
            base_agent = b_agent.replace("_Blind", "")
            display_name = f"{base_agent} (Blind)"
            if b_profit_col in row and not pd.isna(row[b_profit_col]):
                profit = row[b_profit_col]
                if profit < -5000:
                    continue
                rows.append({
                    'Network': network, 'Demand': demand,
                    'Goodwill': goodwill, 'Backlog': backlog,
                    'Agent': display_name, 'Blind': True,
                    'Profit': profit,
                    'FillRate': row.get(f"{b_agent}_FillRate", np.nan),
                    'AvgInv': row.get(f"{b_agent}_AvgInv", np.nan),
                    'Unfulfilled': row.get(f"{b_agent}_Unfulfilled", np.nan),
                })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Chart 01 — Pareto Frontier
# ─────────────────────────────────────────────────────────────────────────────

def plot_pareto_frontier(processed_df, out_dir):
    """Cost-Service Trade-off with Pareto dominance line."""
    base_df = processed_df[~processed_df['Blind']]
    agg_df = base_df.groupby('Agent')[['FillRate', 'Profit']].mean().reset_index()

    agents_present = agg_df['Agent'].tolist()
    palette = get_palette(agents_present)

    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    sns.scatterplot(
        data=agg_df, x='FillRate', y='Profit',
        hue='Agent', hue_order=get_agent_order(agents_present),
        s=200, style='Agent', palette=palette, ax=ax, zorder=5,
    )

    # ── Pareto dominance step-line ──
    pareto_pts = agg_df.sort_values('FillRate', ascending=True).copy()
    pareto_front = []
    max_profit = -np.inf
    for _, r in pareto_pts.iterrows():
        if r['Profit'] >= max_profit:
            pareto_front.append(r)
            max_profit = r['Profit']
    if pareto_front:
        pf = pd.DataFrame(pareto_front).sort_values('FillRate')
        ax.step(pf['FillRate'], pf['Profit'], where='post',
                color='#888', linewidth=0.8, linestyle='--', alpha=0.6, zorder=3)

    # ── Annotate points (with offset to avoid overlap) ──
    offsets = {'Oracle': (0.005, 15), 'MSSP': (0.005, -25),
               'RLGNN': (0.005, 15), 'Residual': (-0.03, -25),
               'RLV4': (0.005, -25)}
    for _, r in agg_df.iterrows():
        dx, dy = offsets.get(r['Agent'], (0.005, 10))
        ax.annotate(r['Agent'], (r['FillRate'], r['Profit']),
                    xytext=(r['FillRate'] + dx, r['Profit'] + dy),
                    fontsize=7, weight='semibold', color='#333')

    ax.set_title('Cost–Service Trade-off (Pareto Frontier)')
    ax.set_xlabel('Average Service Level (Fill Rate)')
    ax.set_ylabel('Average Profit ($)')
    ax.legend(fontsize=7, title='Agent', title_fontsize=7,
              loc='lower right', framealpha=0.9)

    save_fig(fig, os.path.join(out_dir, '01_pareto_frontier'))


# ─────────────────────────────────────────────────────────────────────────────
#  Chart 02 — Algorithm Generalizability
# ─────────────────────────────────────────────────────────────────────────────

def plot_algorithm_generalizability(processed_df, out_dir):
    """Profit across varied Demand Profiles — proves models don't just overfit."""
    base_df = processed_df[~processed_df['Blind']].copy()
    agents = get_agent_order(base_df['Agent'].unique())
    palette = get_palette(agents)

    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    sns.barplot(
        data=base_df, x='Demand', y='Profit', hue='Agent',
        hue_order=agents, order=DEMAND_ORDER,
        errorbar=('ci', 95), palette=palette, ax=ax,
        edgecolor='white', linewidth=0.3,
    )

    ax.set_title('Algorithm Robustness Across Demand Profiles')
    ax.set_xlabel('Demand Regime')
    ax.set_ylabel('Profit ($)')
    ax.legend(fontsize=6, title='Agent', title_fontsize=7,
              bbox_to_anchor=(1.02, 1), loc='upper left', framealpha=0.9)

    save_fig(fig, os.path.join(out_dir, '02_algorithm_generalizability'))


# ─────────────────────────────────────────────────────────────────────────────
#  Chart 03 — Optimality Gap Distribution
# ─────────────────────────────────────────────────────────────────────────────

def plot_optimality_gap(df, out_dir):
    """Violin + strip showing how close each agent is to the Oracle."""
    agents_to_compare = ['MSSP', 'RLV4', 'Residual', 'RLGNN']
    gap_rows = []

    for _, row in df.iterrows():
        oracle_profit = row.get('Oracle_Profit', pd.NA)
        if pd.isna(oracle_profit) or oracle_profit == 0:
            continue
        for agent in agents_to_compare:
            profit_col = f"{agent}_Profit"
            if profit_col not in df.columns or pd.isna(row.get(profit_col)):
                continue
            profit = row[profit_col]
            if profit < -5000:      # skip catastrophic failures
                continue
            gap_pct = ((oracle_profit - profit) / abs(oracle_profit)) * 100
            gap_pct = np.clip(gap_pct, -50, 200)
            gap_rows.append({
                'Agent': agent,
                'Demand': row['Demand'],
                'Network': row['Network'],
                'Gap (%)': gap_pct,
            })

    gap_df = pd.DataFrame(gap_rows)
    agent_order = get_agent_order(gap_df['Agent'].unique())
    palette = get_palette(agent_order)

    fig, ax = plt.subplots(figsize=FIG_DOUBLE)

    # Violin for distribution shape
    sns.violinplot(
        data=gap_df, y='Agent', x='Gap (%)', hue='Agent',
        order=agent_order, hue_order=agent_order,
        palette=palette, inner=None, alpha=0.3, ax=ax, linewidth=0.6,
        legend=False,
    )
    # Strip for individual scenario points
    sns.stripplot(
        data=gap_df, y='Agent', x='Gap (%)', hue='Agent',
        order=agent_order, hue_order=agent_order,
        palette=palette, size=4, alpha=0.7, jitter=0.15, ax=ax,
        legend=False,
    )

    ax.axvline(0, color='#c0392b', linestyle='--', linewidth=1.2, label='Oracle Bound')

    # Median labels
    for agent in agent_order:
        med = gap_df[gap_df['Agent'] == agent]['Gap (%)'].median()
        y_pos = agent_order.index(agent)
        ax.text(med, y_pos - 0.35, f'{med:.1f}%', fontsize=6,
                ha='center', color='#333', weight='bold')

    ax.set_title('Distribution of Relative Optimality Gaps')
    ax.set_xlabel('Gap Below Oracle (%)')
    ax.set_ylabel('')
    ax.legend(fontsize=7, loc='upper right')

    save_fig(fig, os.path.join(out_dir, '03_relative_optimality_gap'))


# ─────────────────────────────────────────────────────────────────────────────
#  Chart 04 — Value of Information
# ─────────────────────────────────────────────────────────────────────────────

def plot_value_of_information(df, out_dir):
    """Classical VOI and VSS metrics by Demand Type."""
    metrics = {'VPI': 'Value of Perfect Info (VPI)',
               'VSS': 'Value of Stochastic Solution (VSS)'}
    available = {k: v for k, v in metrics.items() if k in df.columns}
    if not available:
        return

    melted = df.melt(
        id_vars=['Demand', 'Network'],
        value_vars=list(available.keys()),
        var_name='Metric_Code', value_name='Value ($)',
    )
    melted['Metric'] = melted['Metric_Code'].map(available)

    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    pal = {'Value of Perfect Info (VPI)': '#4c72b0',
           'Value of Stochastic Solution (VSS)': '#55a868'}
    sns.barplot(
        data=melted, x='Demand', y='Value ($)', hue='Metric',
        order=DEMAND_ORDER, errorbar=('ci', 95),
        palette=pal, ax=ax, edgecolor='white', linewidth=0.3,
    )

    ax.set_title('Value of Information & Stochastic Solution')
    ax.set_xlabel('Demand Profile')
    ax.set_ylabel('Economic Value Added ($)')
    ax.legend(fontsize=7, title_fontsize=7, framealpha=0.9)

    save_fig(fig, os.path.join(out_dir, '04_value_of_information'))


# ─────────────────────────────────────────────────────────────────────────────
#  Chart 05 — Information Asymmetry
# ─────────────────────────────────────────────────────────────────────────────

def plot_information_asymmetry(processed_df, out_dir):
    """Aware vs Blind performance with descriptive legend labels."""
    # Find agents that have both aware and blind variants
    blind_agents = processed_df[processed_df['Blind']]['Agent'].str.replace(' (Blind)', '', regex=False).unique()
    if len(blind_agents) == 0:
        return

    filtered = processed_df[
        processed_df['Agent'].str.replace(' (Blind)', '', regex=False).isin(blind_agents)
    ].copy()
    filtered['Base_Agent'] = filtered['Agent'].str.replace(' (Blind)', '', regex=False)
    filtered['Visibility'] = filtered['Blind'].map(BLIND_LABELS)

    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    vis_pal = {'Full Visibility': '#2c7bb6', 'Partial Observability': '#c44e52'}
    sns.barplot(
        data=filtered, x='Base_Agent', y='Profit', hue='Visibility',
        errorbar=('ci', 95), palette=vis_pal, ax=ax,
        edgecolor='white', linewidth=0.3,
    )

    # Add % drop annotations
    for agent in blind_agents:
        aware_vals = filtered[(filtered['Base_Agent'] == agent) & (~filtered['Blind'])]['Profit']
        blind_vals = filtered[(filtered['Base_Agent'] == agent) & (filtered['Blind'])]['Profit']
        if aware_vals.empty or blind_vals.empty:
            continue
        aware_mean = aware_vals.mean()
        blind_mean = blind_vals.mean()
        if aware_mean != 0 and not np.isnan(blind_mean) and not np.isnan(aware_mean):
            pct = ((aware_mean - blind_mean) / abs(aware_mean)) * 100
            x_pos = list(blind_agents).index(agent)
            y_pos = max(aware_mean, blind_mean) + 20
            ax.text(x_pos, y_pos, f'−{pct:.1f}%', ha='center',
                    fontsize=7, color='#c0392b', weight='bold')

    ax.set_title('Impact of Information Asymmetry')
    ax.set_ylabel('Average Profit ($)')
    ax.set_xlabel('Agent')
    ax.legend(fontsize=7, title='Observability', title_fontsize=7, framealpha=0.9)

    save_fig(fig, os.path.join(out_dir, '05_information_asymmetry'))


# ─────────────────────────────────────────────────────────────────────────────
#  Chart 06 — Overall Average Profit
# ─────────────────────────────────────────────────────────────────────────────

def plot_overall_average_profit(processed_df, out_dir):
    """Horizontal bar chart of mean profit with error bars."""
    base_df = processed_df[~processed_df['Blind']]
    agent_means = base_df.groupby('Agent')['Profit'].agg(['mean', 'std']).reset_index()
    agent_means.columns = ['Agent', 'Mean_Profit', 'Std_Profit']
    # Sort by canonical order intersected with data
    order = get_agent_order(agent_means['Agent'].unique())
    agent_means['sort_key'] = agent_means['Agent'].apply(
        lambda a: order.index(a) if a in order else 99)
    agent_means = agent_means.sort_values('sort_key')

    palette = get_palette(agent_means['Agent'])

    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    bars = ax.barh(
        agent_means['Agent'], agent_means['Mean_Profit'],
        xerr=agent_means['Std_Profit'] * 0.5,  # half-std for readability
        color=[palette.get(a, '#666') for a in agent_means['Agent']],
        edgecolor='white', linewidth=0.3, height=0.65,
        capsize=3, ecolor='#666',
    )

    # Value labels
    for bar, val in zip(bars, agent_means['Mean_Profit']):
        ax.text(val - 15, bar.get_y() + bar.get_height() / 2,
                f'${val:.0f}', va='center', ha='right',
                fontsize=7, color='white', weight='bold')

    ax.set_title('Overall Average Profit by Agent')
    ax.set_xlabel('Average Profit ($)')
    ax.set_ylabel('')

    save_fig(fig, os.path.join(out_dir, '06_overall_average_profit'))


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    input_csv = "data/results/benchmark_results_comprehensive_iterative.csv"
    out_dir = "data/results/charts"
    os.makedirs(out_dir, exist_ok=True)

    df = load_data(input_csv)
    if df is None:
        return

    processed_df = process_data(df)
    print(f"Processed {len(processed_df)} agent-scenario records.\n")

    print("1/6  Pareto Frontier ...")
    plot_pareto_frontier(processed_df, out_dir)

    print("2/6  Algorithm Generalizability ...")
    plot_algorithm_generalizability(processed_df, out_dir)

    print("3/6  Relative Optimality Gap ...")
    plot_optimality_gap(df, out_dir)

    print("4/6  Value of Information ...")
    plot_value_of_information(df, out_dir)

    print("5/6  Information Asymmetry ...")
    plot_information_asymmetry(processed_df, out_dir)

    print("6/6  Overall Average Profit ...")
    plot_overall_average_profit(processed_df, out_dir)

    print(f"\n✅ All 6 charts saved to {out_dir}/  (PNG + PDF)")


if __name__ == "__main__":
    main()
