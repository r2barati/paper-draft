"""
generate_speed_vs_quality.py — Speed-Quality Pareto Frontier (chart D).

Compares inference time (ms, log scale) against average profit for each agent.
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


def plot_speed_vs_quality(df, out_dir):
    print("Generating Speed vs Quality Pareto Frontier...")

    agent_map = {
        'MSSP':     ('MSSP_Profit',     'Time_MSSP'),
        'RLV4':     ('RLV4_Profit',     'Time_Sec_RLV4'),
        'Residual': ('Residual_Profit', 'Time_Sec_Residual'),
        'RLGNN':    ('RLGNN_Profit',    'Time_Sec_RLGNN'),
    }

    records = []
    for agent, (p_col, t_col) in agent_map.items():
        if p_col not in df.columns or t_col not in df.columns:
            continue
        valid = df.dropna(subset=[p_col, t_col])
        profit = valid[p_col].mean()
        if profit < -5000:  # skip catastrophic
            continue
        try:
            time_s = valid[t_col].astype(float).mean()
        except Exception:
            time_s = 60.0
        records.append({
            'Agent': agent,
            'Profit': profit,
            'Time_ms': time_s * 1000,
        })

    agg = pd.DataFrame(records)
    if agg.empty:
        print("  No valid timing data.")
        return

    agents = agg['Agent'].tolist()
    palette = get_palette(agents)

    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    sns.scatterplot(
        data=agg, x='Time_ms', y='Profit',
        hue='Agent', hue_order=get_agent_order(agents),
        s=200, style='Agent', palette=palette, ax=ax, zorder=5,
    )

    # Pareto line (lower time, higher profit = better)
    sorted_agg = agg.sort_values('Time_ms')
    pareto = []
    max_p = -np.inf
    for _, r in sorted_agg.iterrows():
        if r['Profit'] >= max_p:
            pareto.append(r)
            max_p = r['Profit']
    if pareto:
        pf = pd.DataFrame(pareto).sort_values('Time_ms')
        ax.step(pf['Time_ms'], pf['Profit'], where='post',
                color='#888', linewidth=0.8, linestyle='--', alpha=0.6, zorder=3)

    # Labels with offsets
    for _, r in agg.iterrows():
        ax.annotate(
            f"{r['Agent']}\n({r['Time_ms']:.0f} ms, ${r['Profit']:.0f})",
            (r['Time_ms'], r['Profit']),
            xytext=(12, -8), textcoords='offset points',
            fontsize=6.5, weight='semibold', color='#333',
        )

    ax.set_xscale('log')
    ax.set_title('Speed–Quality Frontier')
    ax.set_xlabel('Inference Time per Episode (ms) — Log Scale')
    ax.set_ylabel('Average Profit ($)')
    ax.grid(True, which='both', ls='--', alpha=0.3)

    # "Better" corner
    ax.annotate('← Better (Fast & Profitable)', xy=(0.02, 0.95),
                xycoords='axes fraction', fontsize=7, color='#27ae60',
                weight='bold', alpha=0.7)

    ax.legend(fontsize=7, title='Agent', title_fontsize=7, framealpha=0.9)

    save_fig(fig, os.path.join(out_dir, 'D_speed_vs_quality'))


def main():
    input_csv = "data/results/benchmark_results_comprehensive_iterative.csv"
    out_dir = "data/results/charts_advanced"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    plot_speed_vs_quality(df, out_dir)
    print(f"✅ Speed-quality chart saved to {out_dir}/")


if __name__ == "__main__":
    main()
