"""
generate_architecture_hierarchy.py — Architecture Hierarchy chart (F).

Horizontal bar chart for the "hardest scenario" (base | shock | Goodwill=True),
showing how different architectures perform under maximum stress.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from chart_style import (
    apply_style, save_fig, get_palette, get_agent_order, FIG_DOUBLE,
)

apply_style()


def main():
    input_csv = "data/results/benchmark_results_comprehensive_iterative.csv"
    out_dir = "data/results/charts_advanced"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(input_csv)

    # Filter for the hardest scenario
    target = df[
        (df['Network'] == 'base') &
        (df['Demand'] == 'shock') &
        (df['Goodwill'] == True)
    ]

    if target.empty:
        print("WARNING: Could not find (base | shock | Goodwill=True) scenario.")
        return

    agents = ['Oracle', 'MSSP', 'RLGNN', 'Residual', 'RLV4']
    row = target.iloc[0]
    records = []

    for agent in agents:
        val = row.get(f"{agent}_Profit", np.nan)
        std = row.get(f"{agent}_Profit_Std", 0)
        if not pd.isna(val) and val > -5000:
            records.append({'Agent': agent, 'Profit': val, 'Std': std if not pd.isna(std) else 0})

    plot_df = pd.DataFrame(records)
    order = get_agent_order(plot_df['Agent'].unique())
    plot_df['sort'] = plot_df['Agent'].apply(lambda a: order.index(a) if a in order else 99)
    plot_df = plot_df.sort_values('sort')
    palette = get_palette(plot_df['Agent'])

    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    bars = ax.barh(
        plot_df['Agent'], plot_df['Profit'],
        xerr=plot_df['Std'] * 0.5,
        color=[palette.get(a, '#666') for a in plot_df['Agent']],
        edgecolor='white', linewidth=0.3, height=0.6,
        capsize=3, ecolor='#666',
    )

    # Value labels inside bars
    for bar, val in zip(bars, plot_df['Profit']):
        ax.text(val - 15, bar.get_y() + bar.get_height() / 2,
                f'${val:.0f}', va='center', ha='right',
                fontsize=7, color='white', weight='bold')

    # GNN vs MSSP annotation
    if 'RLGNN' in plot_df['Agent'].values and 'MSSP' in plot_df['Agent'].values:
        gnn_p = plot_df[plot_df['Agent'] == 'RLGNN']['Profit'].values[0]
        mssp_p = plot_df[plot_df['Agent'] == 'MSSP']['Profit'].values[0]
        if mssp_p != 0:
            pct = ((gnn_p - mssp_p) / abs(mssp_p)) * 100
            sign = '+' if pct > 0 else ''
            ax.annotate(
                f'GNN vs MSSP: {sign}{pct:.1f}%',
                xy=(mssp_p, 1), xytext=(gnn_p * 0.6, 0.3),
                arrowprops=dict(facecolor='#333', shrink=0.05, width=1),
                fontsize=8, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#666', alpha=0.9),
            )

    ax.set_title('Architecture Hierarchy — Hardest Scenario\n'
                 '(Base Topology · Demand Shock · Goodwill Penalty)')
    ax.set_xlabel('Episode Profit ($)')
    ax.set_ylabel('')

    save_fig(fig, os.path.join(out_dir, 'F_architectural_hierarchy_hardest'))
    print(f"✅ Architecture hierarchy chart saved to {out_dir}/")


if __name__ == "__main__":
    main()
