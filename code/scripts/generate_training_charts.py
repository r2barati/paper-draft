"""
generate_training_charts.py — Charts for the AI / Technical Community.

Answers: "How does the RL training work? Why does architecture matter?"

Charts:
  T1  Learning Curves (from evaluations.npz)
  T2  Training Stability (reward variance over time)
  T3  Observation Space Comparison (visual table)
  T4  Fill Rate Distribution by Agent (violin plot)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from chart_style import (
    apply_style, save_fig, get_palette, get_agent_order,
    FIG_DOUBLE, FIG_WIDE, AGENT_PALETTE,
)

apply_style()


def load_evaluations(log_dir):
    """Parse stable-baselines3 evaluations.npz file."""
    path = os.path.join(log_dir, 'evaluations.npz')
    if not os.path.exists(path):
        return None, None, None
    data = np.load(path)
    timesteps = data['timesteps']
    results = data['results']  # (n_evals, n_envs)
    mean_rewards = np.mean(results, axis=1)
    std_rewards = np.std(results, axis=1)
    return timesteps, mean_rewards, std_rewards


# ─────────────────────────────────────────────────────────────────────────────
#  T1 — Learning Curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(out_dir):
    """Multi-agent learning curves with confidence bands."""
    log_dirs = {
        'PPO-GNN':      ('data/logs/ppo_gnn_logs',      '#1a9641'),
        'PPO-Residual': ('data/logs/ppo_residual_logs',  '#fdae61'),
    }

    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    plotted = False

    for name, (log_dir, color) in log_dirs.items():
        ts, mean_r, std_r = load_evaluations(log_dir)
        if ts is None:
            print(f"  [skip] No evaluations.npz in {log_dir}")
            continue

        # Smooth with rolling window if enough data
        if len(mean_r) > 10:
            window = max(3, len(mean_r) // 20)
            smooth = pd.Series(mean_r).rolling(window, min_periods=1).mean().values
            smooth_std = pd.Series(std_r).rolling(window, min_periods=1).mean().values
        else:
            smooth = mean_r
            smooth_std = std_r

        ax.plot(ts, smooth, label=name, linewidth=1.8, color=color)
        ax.fill_between(ts, smooth - smooth_std, smooth + smooth_std,
                        alpha=0.15, color=color)
        plotted = True

    if not plotted:
        print("  No training data found.")
        plt.close(fig)
        return

    # Format x-axis
    def fmt_k(x, pos):
        if x >= 1_000_000:
            return f'{x/1_000_000:.0f}M'
        elif x >= 1_000:
            return f'{x/1_000:.0f}k'
        return str(int(x))

    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_k))
    ax.set_title('RL Learning Curves — Evaluation Reward Over Training')
    ax.set_xlabel('Training Timesteps')
    ax.set_ylabel('Mean Evaluation Reward')
    ax.legend(fontsize=8, title='Architecture', title_fontsize=8)
    ax.axhline(0, color='#999', linewidth=0.5, linestyle='--', alpha=0.5)

    save_fig(fig, os.path.join(out_dir, 'T1_learning_curves'))


# ─────────────────────────────────────────────────────────────────────────────
#  T2 — Training Stability
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_stability(out_dir):
    """Reward standard deviation over training — shows policy stabilization."""
    log_dirs = {
        'PPO-GNN':      ('data/logs/ppo_gnn_logs',      '#1a9641'),
        'PPO-Residual': ('data/logs/ppo_residual_logs',  '#fdae61'),
    }

    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    plotted = False

    for name, (log_dir, color) in log_dirs.items():
        ts, _, std_r = load_evaluations(log_dir)
        if ts is None:
            continue

        if len(std_r) > 10:
            window = max(3, len(std_r) // 20)
            smooth = pd.Series(std_r).rolling(window, min_periods=1).mean().values
        else:
            smooth = std_r

        ax.plot(ts, smooth, label=name, linewidth=1.8, color=color)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    def fmt_k(x, pos):
        if x >= 1_000_000:
            return f'{x/1_000_000:.0f}M'
        elif x >= 1_000:
            return f'{x/1_000:.0f}k'
        return str(int(x))

    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_k))
    ax.set_title('Training Stability — Reward Variance Over Time')
    ax.set_xlabel('Training Timesteps')
    ax.set_ylabel('Reward Std Dev (lower = more stable)')
    ax.legend(fontsize=8, title='Architecture', title_fontsize=8)

    save_fig(fig, os.path.join(out_dir, 'T2_training_stability'))


# ─────────────────────────────────────────────────────────────────────────────
#  T3 — Observation Space Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_observation_comparison(out_dir):
    """Visual table comparing what each architecture 'sees'."""
    architectures = [
        {
            'name': 'RLV1 (Baseline PPO)',
            'obs_dim': '~25+',
            'features': 'Raw state: demand, all\ninventory, all pipeline,\ntimestep, sentiment',
            'structure': 'Flat MLP',
            'color': '#bdbdbd',
        },
        {
            'name': 'RLV4 (Pruned PPO)',
            'obs_dim': '~12',
            'features': 'Domain-pruned: normalized\ndemand/inv/pipeline ratios,\nsin/cos time encoding',
            'structure': 'Flat MLP',
            'color': '#2b8cbe',
        },
        {
            'name': 'RLGNN',
            'obs_dim': '5 per node',
            'features': 'Graph-structured: per-node\ninventory, demand, capacity,\ntemporal encoding',
            'structure': 'MPNN + Edge-conditioned\nMessage Passing',
            'color': '#1a9641',
        },
        {
            'name': 'Residual RL',
            'obs_dim': '~12 + heuristic',
            'features': 'Same as RLV4 + heuristic\nbase action; learns Δ only',
            'structure': 'Flat MLP\n(additive residual)',
            'color': '#fdae61',
        },
    ]

    fig, ax = plt.subplots(figsize=FIG_WIDE)
    ax.axis('off')

    # Table headers
    headers = ['Architecture', 'Obs Dim', 'Input Features', 'Network Structure']
    col_widths = [0.22, 0.1, 0.38, 0.30]

    # Draw header
    y_start = 0.92
    x_pos = 0.02
    for i, (h, w) in enumerate(zip(headers, col_widths)):
        ax.text(x_pos + w / 2, y_start, h, ha='center', va='center',
                fontsize=9, weight='bold', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', fc='#333', ec='none', alpha=0.9),
                color='white')
        x_pos += w

    # Draw rows
    row_height = 0.18
    for r_idx, arch in enumerate(architectures):
        y = y_start - (r_idx + 1) * row_height - 0.03
        x_pos = 0.02

        values = [arch['name'], arch['obs_dim'], arch['features'], arch['structure']]
        for c_idx, (val, w) in enumerate(zip(values, col_widths)):
            bg_color = arch['color'] if c_idx == 0 else '#f7f7f7'
            fg_color = 'white' if c_idx == 0 else '#333'
            fontsize = 8 if c_idx == 0 else 7

            ax.text(x_pos + w / 2, y, val, ha='center', va='center',
                    fontsize=fontsize, transform=ax.transAxes, color=fg_color,
                    bbox=dict(boxstyle='round,pad=0.3', fc=bg_color,
                              ec='#ddd', alpha=0.85),
                    linespacing=1.4)
            x_pos += w

    ax.set_title('Observation Space Comparison Across Architectures',
                 fontsize=11, pad=20)

    save_fig(fig, os.path.join(out_dir, 'T3_observation_comparison'))


# ─────────────────────────────────────────────────────────────────────────────
#  T4 — Fill Rate Distribution by Agent
# ─────────────────────────────────────────────────────────────────────────────

def plot_fill_rate_distribution(df, out_dir):
    """Violin plot of fill rates across all scenarios."""
    agents = ['Oracle', 'RLGNN', 'MSSP', 'Residual', 'RLV4']
    records = []

    for _, row in df.iterrows():
        for agent in agents:
            fr_col = f'{agent}_FillRate'
            p_col = f'{agent}_Profit'
            if fr_col in df.columns and not pd.isna(row.get(fr_col)):
                # Skip catastrophic failures
                profit = row.get(p_col, 0)
                if not pd.isna(profit) and profit < -5000:
                    continue
                records.append({
                    'Agent': agent,
                    'Fill Rate': row[fr_col],
                    'Demand': row['Demand'],
                })

    fr_df = pd.DataFrame(records)
    agent_order = get_agent_order(fr_df['Agent'].unique())
    palette = get_palette(agent_order)

    fig, ax = plt.subplots(figsize=FIG_DOUBLE)

    sns.violinplot(
        data=fr_df, x='Agent', y='Fill Rate', hue='Agent',
        order=agent_order, hue_order=agent_order,
        palette=palette, inner='quartile', ax=ax,
        linewidth=0.6, legend=False,
    )
    sns.stripplot(
        data=fr_df, x='Agent', y='Fill Rate', hue='Agent',
        order=agent_order, hue_order=agent_order,
        palette=palette, size=3, alpha=0.5, jitter=0.15, ax=ax,
        legend=False,
    )

    # Median labels
    for i, agent in enumerate(agent_order):
        med = fr_df[fr_df['Agent'] == agent]['Fill Rate'].median()
        ax.text(i, med + 0.015, f'{med:.2f}', ha='center',
                fontsize=7, color='#333', weight='bold')

    ax.set_title('Service Level (Fill Rate) Distribution by Agent')
    ax.set_ylabel('Fill Rate (higher = better)')
    ax.set_xlabel('')
    ax.set_ylim(0.5, 1.05)
    ax.axhline(1.0, color='#27ae60', linewidth=0.8, linestyle='--',
               alpha=0.5, label='Perfect Service')
    ax.legend(fontsize=7)

    save_fig(fig, os.path.join(out_dir, 'T4_fill_rate_distribution'))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    input_csv = "data/results/benchmark_results_comprehensive_iterative.csv"
    out_dir = "data/results/charts_training"
    os.makedirs(out_dir, exist_ok=True)

    print("T1  Learning Curves ...")
    plot_learning_curves(out_dir)

    print("T2  Training Stability ...")
    plot_training_stability(out_dir)

    print("T3  Observation Space Comparison ...")
    plot_observation_comparison(out_dir)

    df = pd.read_csv(input_csv)

    print("T4  Fill Rate Distribution ...")
    plot_fill_rate_distribution(df, out_dir)

    print(f"\n✅ Training charts saved to {out_dir}/")


if __name__ == "__main__":
    main()
