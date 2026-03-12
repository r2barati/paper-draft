"""
generate_business_charts.py — Charts for Business Decision-Makers.

Answers: "Which model should I deploy for MY supply chain?"

Charts:
  B1  Scenario-Level Performance Heatmap (Agents × Scenarios)
  B2  Model Recommendation Matrix (best model per condition)
  B3  Goodwill Penalty Impact (paired bars)
  B4  Cost-of-Suboptimality Dashboard (dollar loss vs Oracle)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from chart_style import (
    apply_style, AGENT_PALETTE, save_fig, get_palette,
    get_agent_order, FIG_DOUBLE, FIG_WIDE,
)

apply_style()


def load_data(csv_path):
    print(f"Loading data from {csv_path}...")
    return pd.read_csv(csv_path)


# ─────────────────────────────────────────────────────────────────────────────
#  B1 — Scenario-Level Performance Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_scenario_heatmap(df, out_dir):
    """Annotated heatmap: rows = scenarios, columns = agents, colored by opt. gap."""
    agents = ['Oracle', 'RLGNN', 'MSSP', 'Residual', 'RLV4']

    # Build scenario labels and extract profits
    rows = []
    for _, row in df.iterrows():
        gw = 'GW' if row['Goodwill'] else 'NoGW'
        bl = 'BL' if row['Backlog'] else 'LS'
        label = f"{row['Network']} · {row['Demand']} · {gw} · {bl}"
        oracle_p = row.get('Oracle_Profit', np.nan)

        entry = {'Scenario': label}
        for agent in agents:
            p = row.get(f'{agent}_Profit', np.nan)
            if not pd.isna(p) and p > -5000:
                entry[agent] = p
                if not pd.isna(oracle_p) and oracle_p != 0:
                    entry[f'{agent}_gap'] = ((oracle_p - p) / abs(oracle_p)) * 100
                else:
                    entry[f'{agent}_gap'] = 0
            else:
                entry[agent] = np.nan
                entry[f'{agent}_gap'] = np.nan
        rows.append(entry)

    heatmap_df = pd.DataFrame(rows).set_index('Scenario')
    profit_matrix = heatmap_df[agents]
    gap_matrix = heatmap_df[[f'{a}_gap' for a in agents]]
    gap_matrix.columns = agents

    # Annotate: show profit values, color by gap
    fig, ax = plt.subplots(figsize=(10, max(8, len(heatmap_df) * 0.35)))

    # Cap gap for color mapping
    gap_capped = gap_matrix.clip(-20, 100)

    sns.heatmap(
        gap_capped, annot=profit_matrix, fmt='.0f',
        cmap='RdYlGn_r', center=20,
        linewidths=0.5, linecolor='white',
        ax=ax, cbar_kws={'label': 'Optimality Gap (%)'},
        annot_kws={'fontsize': 6.5},
    )

    # Bold the best (non-Oracle) agent per row
    for i, (_, row) in enumerate(profit_matrix.iterrows()):
        non_oracle = row.drop('Oracle', errors='ignore').dropna()
        if not non_oracle.empty:
            best_agent = non_oracle.idxmax()
            j = agents.index(best_agent)
            ax.add_patch(mpl.patches.Rectangle(
                (j, i), 1, 1, fill=False, edgecolor='#1a9641',
                linewidth=2.5, zorder=10))

    ax.set_title('Scenario-Level Performance Heatmap\n'
                 '(Values = Profit $, Color = Gap from Oracle, Green Border = Best Non-Oracle)',
                 fontsize=10, pad=15)
    ax.set_ylabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6.5)

    save_fig(fig, os.path.join(out_dir, 'B1_scenario_heatmap'))


# ─────────────────────────────────────────────────────────────────────────────
#  B2 — Model Recommendation Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_recommendation_matrix(df, out_dir):
    """For each environmental condition, show which non-Oracle model wins."""
    agents = ['RLGNN', 'MSSP', 'Residual', 'RLV4']

    conditions = [
        ('By Demand', 'Demand', df['Demand'].unique()),
        ('By Topology', 'Network', df['Network'].unique()),
        ('By Goodwill', 'Goodwill', df['Goodwill'].unique()),
        ('By Backlog', 'Backlog', df['Backlog'].unique()),
    ]

    fig, axes = plt.subplots(2, 2, figsize=FIG_WIDE)
    axes = axes.flatten()

    for idx, (title, col, values) in enumerate(conditions):
        ax = axes[idx]
        records = []
        for val in sorted(values, key=str):
            subset = df[df[col] == val]
            label = str(val)
            if col == 'Goodwill':
                label = 'With Goodwill' if val else 'No Goodwill'
            elif col == 'Backlog':
                label = 'With Backlog' if val else 'Lost Sales'

            for agent in agents:
                p_col = f'{agent}_Profit'
                if p_col in subset.columns:
                    vals = subset[p_col].dropna()
                    vals = vals[vals > -5000]
                    if not vals.empty:
                        records.append({
                            'Condition': label,
                            'Agent': agent,
                            'Profit': vals.mean(),
                        })

        plot_df = pd.DataFrame(records)
        if plot_df.empty:
            continue

        palette = get_palette(agents)
        sns.barplot(
            data=plot_df, x='Condition', y='Profit', hue='Agent',
            hue_order=get_agent_order(plot_df['Agent'].unique()),
            palette=palette, ax=ax, edgecolor='white', linewidth=0.3,
        )

        # Star the best agent per condition
        for cond in plot_df['Condition'].unique():
            cond_data = plot_df[plot_df['Condition'] == cond]
            if not cond_data.empty:
                best = cond_data.loc[cond_data['Profit'].idxmax()]
                x_idx = list(sorted(plot_df['Condition'].unique(), key=str)).index(cond)
                ax.text(x_idx, best['Profit'] + 15, 'BEST',
                        ha='center', fontsize=5.5, color='#1a9641',
                        weight='bold')

        ax.set_title(title, fontsize=9, weight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Profit ($)' if idx % 2 == 0 else '')
        ax.legend(fontsize=5.5, title_fontsize=6, loc='lower right')
        ax.tick_params(labelsize=7)

    fig.suptitle('Model Recommendation by Operating Condition\n(★ = Best Agent)',
                 fontsize=11, y=1.02)
    fig.tight_layout()

    save_fig(fig, os.path.join(out_dir, 'B2_recommendation_matrix'))


# ─────────────────────────────────────────────────────────────────────────────
#  B3 — Goodwill Penalty Impact
# ─────────────────────────────────────────────────────────────────────────────

def plot_goodwill_impact(df, out_dir):
    """Paired bars: performance with vs without goodwill penalty."""
    agents = ['Oracle', 'RLGNN', 'MSSP', 'Residual', 'RLV4']
    records = []

    for gw_val in [False, True]:
        subset = df[df['Goodwill'] == gw_val]
        for agent in agents:
            col = f'{agent}_Profit'
            if col in subset.columns:
                vals = subset[col].dropna()
                vals = vals[vals > -5000]
                if not vals.empty:
                    records.append({
                        'Agent': agent,
                        'Goodwill': 'With Goodwill' if gw_val else 'No Goodwill',
                        'Profit': vals.mean(),
                    })

    plot_df = pd.DataFrame(records)
    agent_order = get_agent_order(plot_df['Agent'].unique())

    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    gw_pal = {'No Goodwill': '#4c72b0', 'With Goodwill': '#c44e52'}
    sns.barplot(
        data=plot_df, x='Agent', y='Profit', hue='Goodwill',
        order=agent_order, palette=gw_pal, ax=ax,
        edgecolor='white', linewidth=0.3,
    )

    # Add % impact annotations
    for agent in agent_order:
        no_gw = plot_df[(plot_df['Agent'] == agent) & (plot_df['Goodwill'] == 'No Goodwill')]
        with_gw = plot_df[(plot_df['Agent'] == agent) & (plot_df['Goodwill'] == 'With Goodwill')]
        if no_gw.empty or with_gw.empty:
            continue
        no_val = no_gw['Profit'].values[0]
        gw_val = with_gw['Profit'].values[0]
        if no_val != 0:
            pct = ((gw_val - no_val) / abs(no_val)) * 100
            sign = '+' if pct > 0 else ''
            x_pos = agent_order.index(agent)
            y_pos = max(no_val, gw_val) + 25
            ax.text(x_pos, y_pos, f'{sign}{pct:.0f}%', ha='center',
                    fontsize=7, color='#c0392b' if pct < 0 else '#27ae60',
                    weight='bold')

    ax.set_title('Impact of Goodwill (Customer Loyalty) Penalty')
    ax.set_ylabel('Mean Profit ($)')
    ax.set_xlabel('')
    ax.legend(fontsize=7, title='Condition', title_fontsize=7)

    save_fig(fig, os.path.join(out_dir, 'B3_goodwill_impact'))


# ─────────────────────────────────────────────────────────────────────────────
#  B4 — Cost-of-Suboptimality Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def plot_cost_of_suboptimality(df, out_dir):
    """Dollar profit lost vs Oracle for each agent."""
    agents = ['RLGNN', 'MSSP', 'Residual', 'RLV4', 'Heuristic']
    records = []

    for _, row in df.iterrows():
        oracle_p = row.get('Oracle_Profit', np.nan)
        if pd.isna(oracle_p):
            continue
        for agent in agents:
            p = row.get(f'{agent}_Profit', np.nan)
            if pd.isna(p) or p < -5000:
                continue
            records.append({
                'Agent': agent,
                'Dollar_Loss': oracle_p - p,
            })

    loss_df = pd.DataFrame(records)
    agg = loss_df.groupby('Agent')['Dollar_Loss'].agg(['mean', 'std']).reset_index()
    agg.columns = ['Agent', 'Mean_Loss', 'Std_Loss']
    agg = agg.sort_values('Mean_Loss')

    palette = get_palette(agg['Agent'])

    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    bars = ax.barh(
        agg['Agent'], agg['Mean_Loss'],
        xerr=agg['Std_Loss'] * 0.5,
        color=[palette.get(a, '#666') for a in agg['Agent']],
        edgecolor='white', linewidth=0.3, height=0.6,
        capsize=3, ecolor='#666',
    )

    for bar, val in zip(bars, agg['Mean_Loss']):
        if val > 20:
            ax.text(val - 5, bar.get_y() + bar.get_height() / 2,
                    f'${val:.0f}', va='center', ha='right',
                    fontsize=7, color='white', weight='bold')
        else:
            ax.text(val + 5, bar.get_y() + bar.get_height() / 2,
                    f'${val:.0f}', va='center', ha='left',
                    fontsize=7, color='#333', weight='bold')

    ax.set_title('Cost of Suboptimality\n(Average Profit Lost vs. Oracle per Episode)')
    ax.set_xlabel('Profit Lost vs. Oracle ($)')
    ax.set_ylabel('')
    ax.axvline(0, color='#333', linewidth=0.5)

    save_fig(fig, os.path.join(out_dir, 'B4_cost_of_suboptimality'))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    input_csv = "data/results/benchmark_results_comprehensive_iterative.csv"
    out_dir = "data/results/charts_business"
    os.makedirs(out_dir, exist_ok=True)

    df = load_data(input_csv)
    if df is None:
        return

    print("B1  Scenario Heatmap ...")
    plot_scenario_heatmap(df, out_dir)

    print("B2  Recommendation Matrix ...")
    plot_recommendation_matrix(df, out_dir)

    print("B3  Goodwill Impact ...")
    plot_goodwill_impact(df, out_dir)

    print("B4  Cost of Suboptimality ...")
    plot_cost_of_suboptimality(df, out_dir)

    print(f"\n✅ Business charts saved to {out_dir}/")


if __name__ == "__main__":
    main()
