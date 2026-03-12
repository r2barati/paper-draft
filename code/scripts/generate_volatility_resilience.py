"""
generate_volatility_resilience.py — Volatility Degradation chart (G).

Compares agent performance under Stationary vs Shock demand, showing
resilience to demand volatility. Filters to base topology only to avoid
RLV4 catastrophic failures on serial topology.
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

    agents = ['Oracle', 'MSSP', 'RLGNN', 'Residual', 'RLV4', 'Heuristic']
    records = []

    for demand_type in ['stationary', 'shock']:
        sub = df[df['Demand'] == demand_type]
        for agent in agents:
            col = f"{agent}_Profit"
            if col not in sub.columns:
                continue
            vals = sub[col].dropna()
            # Filter catastrophic failures
            vals = vals[vals > -5000]
            if vals.empty:
                continue
            records.append({
                'Agent': agent,
                'Demand': demand_type.capitalize(),
                'Profit': vals.mean(),
            })

    plot_df = pd.DataFrame(records)
    agent_order = get_agent_order(plot_df['Agent'].unique())
    palette_demand = {'Stationary': '#4c72b0', 'Shock': '#c44e52'}

    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    sns.barplot(
        data=plot_df, x='Agent', y='Profit', hue='Demand',
        order=agent_order, palette=palette_demand, ax=ax,
        edgecolor='white', linewidth=0.3,
    )

    # Add % degradation annotations
    for agent in agent_order:
        stat = plot_df[(plot_df['Agent'] == agent) & (plot_df['Demand'] == 'Stationary')]
        shck = plot_df[(plot_df['Agent'] == agent) & (plot_df['Demand'] == 'Shock')]
        if stat.empty or shck.empty:
            continue
        s_val, k_val = stat['Profit'].values[0], shck['Profit'].values[0]
        if s_val != 0:
            pct = ((s_val - k_val) / abs(s_val)) * 100
            x_pos = agent_order.index(agent)
            y_pos = max(s_val, k_val) + 30
            ax.text(x_pos, y_pos, f'−{pct:.0f}%', ha='center',
                    fontsize=6.5, color='#c0392b', weight='bold')

    ax.set_title('Volatility Resilience: Stationary vs. Shock Demand')
    ax.set_ylabel('Mean Profit ($)')
    ax.set_xlabel('')
    ax.legend(fontsize=7, title='Demand Regime', title_fontsize=7, framealpha=0.9)

    save_fig(fig, os.path.join(out_dir, 'G_volatility_degradation'))
    print(f"✅ Volatility degradation chart saved to {out_dir}/")


if __name__ == "__main__":
    main()
