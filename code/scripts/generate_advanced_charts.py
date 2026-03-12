"""
generate_advanced_charts.py — Publication-quality advanced benchmark charts.

Generates 3 advanced analytical charts:
  A  Topology-specific Optimality Gaps
  B  Inventory vs Unfulfilled Demand Trade-off
  C  Unfulfilled Demand Breakdowns by Demand Type
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from chart_style import (
    apply_style, AGENT_PALETTE, DEMAND_ORDER,
    save_fig, get_palette, get_agent_order, FIG_DOUBLE,
)

apply_style()

# ── Data helpers ──────────────────────────────────────────────────────────────

def load_data(csv_path):
    print(f"Loading data from {csv_path}...")
    return pd.read_csv(csv_path)


def process_data(df):
    valid_agents = ['Oracle', 'MSSP', 'RLV4', 'Residual', 'RLGNN']
    rows = []
    for _, row in df.iterrows():
        for agent in valid_agents:
            profit_col = f"{agent}_Profit"
            if profit_col in row and not pd.isna(row[profit_col]):
                profit = row[profit_col]
                if profit < -5000:  # skip catastrophic failures
                    continue
                rows.append({
                    'Network': row['Network'], 'Demand': row['Demand'],
                    'Goodwill': row['Goodwill'], 'Backlog': row['Backlog'],
                    'Agent': agent, 'Blind': False,
                    'Profit': profit,
                    'FillRate': row.get(f"{agent}_FillRate", np.nan),
                    'AvgInv': row.get(f"{agent}_AvgInv", np.nan),
                    'Unfulfilled': row.get(f"{agent}_Unfulfilled", np.nan),
                })
    return pd.DataFrame(rows)


# ── Chart A: Topology Optimality Gaps ─────────────────────────────────────────

def plot_topology_gaps(df, out_dir):
    """Optimality gaps split by Network Topology — violin + strip."""
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
            if profit < -5000:
                continue
            gap = ((oracle_profit - profit) / abs(oracle_profit)) * 100
            gap = np.clip(gap, -50, 200)
            gap_rows.append({
                'Agent': agent, 'Network': row['Network'],
                'Gap (%)': gap,
            })

    gap_df = pd.DataFrame(gap_rows)
    agent_order = get_agent_order(gap_df['Agent'].unique())

    fig, ax = plt.subplots(figsize=FIG_DOUBLE)

    # Use Network as hue
    net_pal = {'base': '#4c72b0', 'serial': '#55a868'}
    sns.violinplot(
        data=gap_df, y='Agent', x='Gap (%)', hue='Network',
        order=agent_order, palette=net_pal,
        inner=None, alpha=0.25, ax=ax, linewidth=0.5,
    )
    sns.stripplot(
        data=gap_df, y='Agent', x='Gap (%)', hue='Network',
        order=agent_order, palette=net_pal,
        dodge=True, size=4, alpha=0.7, jitter=0.12, ax=ax,
    )

    ax.axvline(0, color='#c0392b', linestyle='--', linewidth=1, label='Oracle')
    ax.set_title('Optimality Gaps by Supply Chain Topology')
    ax.set_xlabel('Gap Below Oracle (%)')
    ax.set_ylabel('')

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=7, title='Topology',
              title_fontsize=7)

    save_fig(fig, os.path.join(out_dir, 'A_topology_optimality_gaps'))


# ── Chart B: Inventory vs Unfulfilled Trade-off ──────────────────────────────

def plot_inventory_vs_unfulfilled(processed_df, out_dir):
    """Scatter showing operational trade-off with Pareto line."""
    agg_df = processed_df.groupby('Agent')[['AvgInv', 'Unfulfilled']].mean().reset_index()
    agents = agg_df['Agent'].tolist()
    palette = get_palette(agents)

    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    sns.scatterplot(
        data=agg_df, x='AvgInv', y='Unfulfilled',
        hue='Agent', hue_order=get_agent_order(agents),
        s=200, style='Agent', palette=palette, ax=ax, zorder=5,
    )

    # Pareto frontier line (lower-left is better on both axes)
    sorted_pts = agg_df.sort_values('AvgInv')
    pareto = []
    min_unf = np.inf
    for _, r in sorted_pts.iterrows():
        if r['Unfulfilled'] <= min_unf:
            pareto.append(r)
            min_unf = r['Unfulfilled']
    if pareto:
        pf = pd.DataFrame(pareto).sort_values('AvgInv')
        ax.plot(pf['AvgInv'], pf['Unfulfilled'],
                color='#888', linewidth=0.8, linestyle='--', alpha=0.6, zorder=3)

    # Labels with offsets
    offsets = {'Oracle': (8, -8), 'MSSP': (8, 5), 'RLGNN': (8, -8),
               'Residual': (8, 5), 'RLV4': (8, 5)}
    for _, r in agg_df.iterrows():
        dx, dy = offsets.get(r['Agent'], (8, 3))
        ax.annotate(r['Agent'], (r['AvgInv'], r['Unfulfilled']),
                    xytext=(r['AvgInv'] + dx, r['Unfulfilled'] + dy),
                    fontsize=7, weight='semibold', color='#333')

    # "Better" corner annotation
    ax.annotate('← Better', xy=(0.05, 0.05), xycoords='axes fraction',
                fontsize=8, color='#27ae60', weight='bold', alpha=0.7)

    ax.set_title('Operational Trade-off: Inventory vs. Lost Sales')
    ax.set_xlabel('Average System Inventory (Units)')
    ax.set_ylabel('Average Unfulfilled Demand (Units)')
    ax.legend(fontsize=7, title='Agent', title_fontsize=7, framealpha=0.9)

    save_fig(fig, os.path.join(out_dir, 'B_inv_vs_unfulfilled_tradeoff'))


# ── Chart C: Unfulfilled Demand Breakdowns ───────────────────────────────────

def plot_kpi_breakdown(processed_df, out_dir):
    """Unfulfilled demand by shock type — focused on competitive agents."""
    agents = get_agent_order(processed_df['Agent'].unique())
    palette = get_palette(agents)

    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    sns.barplot(
        data=processed_df, x='Demand', y='Unfulfilled',
        hue='Agent', hue_order=agents, order=DEMAND_ORDER,
        errorbar=None, palette=palette, ax=ax,
        edgecolor='white', linewidth=0.3,
    )

    ax.set_title('Unfulfilled Demand by Demand Regime')
    ax.set_ylabel('Mean Unfulfilled Units')
    ax.set_xlabel('Demand Type')
    ax.legend(fontsize=6, title='Agent', title_fontsize=7,
              bbox_to_anchor=(1.02, 1), loc='upper left', framealpha=0.9)

    save_fig(fig, os.path.join(out_dir, 'C_unfulfilled_breakdowns'))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    input_csv = "data/results/benchmark_results_comprehensive_iterative.csv"
    out_dir = "data/results/charts_advanced"
    os.makedirs(out_dir, exist_ok=True)

    df = load_data(input_csv)
    if df is None:
        return

    processed_df = process_data(df)
    print(f"Processed {len(processed_df)} records.\n")

    print("A  Topology Gaps ...")
    plot_topology_gaps(df, out_dir)

    print("B  Inventory vs Unfulfilled ...")
    plot_inventory_vs_unfulfilled(processed_df, out_dir)

    print("C  Unfulfilled Breakdowns ...")
    plot_kpi_breakdown(processed_df, out_dir)

    print(f"\n✅ Advanced charts saved to {out_dir}/")


if __name__ == "__main__":
    main()
