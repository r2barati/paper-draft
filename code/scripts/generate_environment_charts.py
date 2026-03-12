"""
generate_environment_charts.py — Charts for Environment & Domain Understanding.

Answers: "What makes this benchmark challenging? What are the dynamics?"

Charts:
  E1  Demand Profile Gallery (2×2 panel with 4 demand regimes)
  E2  Topology Comparison Infographic (base vs serial networks)
  E3  Scenario Complexity Landscape (Oracle profit variance heatmap)
  E4  Reward Decomposition Waterfall (revenue - costs = profit, per agent)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx

from chart_style import apply_style, save_fig, get_palette, FIG_DOUBLE, FIG_WIDE

apply_style()


# ─────────────────────────────────────────────────────────────────────────────
#  E1 — Demand Profile Gallery
# ─────────────────────────────────────────────────────────────────────────────

def plot_demand_gallery(out_dir):
    """2×2 panel showing demand traces for 4 regimes."""
    from src.envs.core.demand_engine import DemandEngine
    from scipy.stats import poisson

    configs = {
        'Stationary': {'type': 'stationary', 'base_mu': 20},
        'Trend':      {'type': 'trend',      'base_mu': 20, 'trend_slope': 0.05},
        'Seasonal':   {'type': 'seasonal',   'base_mu': 20, 'seasonal_amp': 0.5},
        'Shock':      {'type': 'shock',       'base_mu': 20, 'shock_time': 15, 'shock_mag': 2.0},
    }
    colors = {'Stationary': '#4c72b0', 'Trend': '#55a868',
              'Seasonal': '#c44e52', 'Shock': '#8172b2'}

    T = 30
    n_seeds = 15

    fig, axes = plt.subplots(2, 2, figsize=FIG_WIDE, sharex=True)
    axes = axes.flatten()

    for idx, (name, cfg) in enumerate(configs.items()):
        ax = axes[idx]
        engine = DemandEngine(cfg)

        # Generate multiple traces
        all_traces = np.zeros((n_seeds, T))
        mu_trace = np.zeros(T)

        for seed in range(n_seeds):
            rng = np.random.default_rng(seed=seed * 42)
            engine.reset()
            for t in range(T):
                mu = engine.get_current_mu(t)
                mu_trace[t] = mu
                demand = poisson.rvs(mu, random_state=rng)
                all_traces[seed, t] = demand

        # Plot individual traces (faint)
        for s in range(n_seeds):
            ax.plot(range(T), all_traces[s], color=colors[name],
                    alpha=0.15, linewidth=0.6)

        # Plot mean ± std band
        mean_d = all_traces.mean(axis=0)
        std_d = all_traces.std(axis=0)
        ax.plot(range(T), mean_d, color=colors[name], linewidth=1.8, label='Mean')
        ax.fill_between(range(T), mean_d - std_d, mean_d + std_d,
                        alpha=0.2, color=colors[name])

        # Plot deterministic mu
        ax.plot(range(T), mu_trace, color='#333', linewidth=0.8,
                linestyle='--', alpha=0.5, label='μ(t)')

        ax.set_title(name, fontsize=10, weight='bold')
        ax.set_ylabel('Demand (units)' if idx % 2 == 0 else '')
        if idx >= 2:
            ax.set_xlabel('Time Period')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=6, loc='upper left')

        # Annotate key features
        if name == 'Shock':
            ax.axvline(15, color='red', linestyle=':', linewidth=1, alpha=0.7)
            ax.text(15.5, ax.get_ylim()[1] * 0.9, 'Shock\nat t=15',
                    fontsize=6, color='red', weight='bold')
        if name == 'Seasonal':
            ax.text(7, ax.get_ylim()[1] * 0.85, 'Peak',
                    fontsize=6, color=colors[name], weight='bold', ha='center')
            ax.text(22, 5, 'Trough', fontsize=6,
                    color=colors[name], weight='bold', ha='center')

    fig.suptitle('Demand Regime Gallery — 4 Non-Stationary Profiles',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'E1_demand_profile_gallery'))


# ─────────────────────────────────────────────────────────────────────────────
#  E2 — Topology Comparison Infographic
# ─────────────────────────────────────────────────────────────────────────────

def plot_topology_comparison(out_dir):
    """Side-by-side network diagrams for base and serial topologies."""
    from src.envs.core.network_topology import SupplyChainNetwork

    fig, axes = plt.subplots(1, 2, figsize=FIG_WIDE)

    for idx, (scenario, title) in enumerate([('base', 'Base Topology\n(9 nodes, 12 edges)'),
                                              ('serial', 'Serial Topology\n(5 nodes, 4 edges)')]):
        ax = axes[idx]
        net = SupplyChainNetwork(scenario=scenario)
        G = net.graph

        # Color by node type
        node_colors = []
        node_labels = {}
        for n in G.nodes():
            if n in net.rawmat:
                node_colors.append('#abd9e9')
                node_labels[n] = f'RM {n}'
            elif n in net.factory:
                node_colors.append('#fdae61')
                cap = G.nodes[n].get('C', '?')
                node_labels[n] = f'Mfg {n}\nC={cap}'
            elif n in net.distrib:
                node_colors.append('#1a9641')
                inv = G.nodes[n].get('I0', '?')
                node_labels[n] = f'Dist {n}\nI₀={inv}'
            elif n in net.market:
                node_colors.append('#d7191c')
                node_labels[n] = f'Mkt {n}'
            else:
                node_colors.append('#999')
                node_labels[n] = str(n)

        # Layout
        if scenario == 'base':
            pos = nx.spring_layout(G, seed=42, k=2)
        else:
            # Linear layout for serial chain
            pos = {n: (i, 0) for i, n in enumerate(sorted(G.nodes()))}

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=800, edgecolors='#333', linewidths=1)
        nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax,
                                font_size=5.5, font_weight='bold')

        # Edge labels (lead times)
        edge_labels = {}
        for u, v, d in G.edges(data=True):
            parts = []
            if 'L' in d:
                parts.append(f"L={d['L']}")
            if 'p' in d:
                parts.append(f"p={d['p']}")
            edge_labels[(u, v)] = '\n'.join(parts)

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#666',
                               arrows=True, arrowsize=12, width=1.2)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     ax=ax, font_size=4.5, alpha=0.8)

        ax.set_title(title, fontsize=10, weight='bold')
        ax.axis('off')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#abd9e9', edgecolor='#333', label='Raw Material'),
        mpatches.Patch(facecolor='#fdae61', edgecolor='#333', label='Manufacturer'),
        mpatches.Patch(facecolor='#1a9641', edgecolor='#333', label='Distributor'),
        mpatches.Patch(facecolor='#d7191c', edgecolor='#333', label='Market'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=7, framealpha=0.9)

    fig.suptitle('Supply Chain Network Topologies', fontsize=11, y=1.01)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig(fig, os.path.join(out_dir, 'E2_topology_comparison'))


# ─────────────────────────────────────────────────────────────────────────────
#  E3 — Scenario Complexity Landscape
# ─────────────────────────────────────────────────────────────────────────────

def plot_complexity_landscape(df, out_dir):
    """Heatmap of Oracle profit across all 32 scenarios."""
    # Build label columns
    df = df.copy()
    df['Row'] = df['Network'] + ' · ' + df['Demand']
    df['Col'] = df.apply(
        lambda r: ('GW' if r['Goodwill'] else 'NoGW') + ' · ' +
                  ('BL' if r['Backlog'] else 'LS'), axis=1)

    pivot = df.pivot_table(values='Oracle_Profit', index='Row', columns='Col',
                           aggfunc='mean')

    fig, ax = plt.subplots(figsize=FIG_DOUBLE)
    sns.heatmap(
        pivot, annot=True, fmt='.0f', cmap='YlOrRd_r',
        linewidths=0.5, linecolor='white', ax=ax,
        cbar_kws={'label': 'Oracle Profit ($)'},
        annot_kws={'fontsize': 8},
    )

    ax.set_title('Scenario Complexity Landscape\n'
                 '(Oracle Profit = Upper Bound — Lower = Harder)')
    ax.set_ylabel('')
    ax.set_xlabel('')

    save_fig(fig, os.path.join(out_dir, 'E3_complexity_landscape'))


# ─────────────────────────────────────────────────────────────────────────────
#  E4 — Reward Decomposition Waterfall
# ─────────────────────────────────────────────────────────────────────────────

def plot_reward_decomposition(df, out_dir):
    """Stacked bar showing reward components: Profit, AvgInv (proxy for holding),
    Unfulfilled (proxy for penalty), and Fill Rate relationship."""
    agents = ['Oracle', 'RLGNN', 'MSSP', 'Residual']
    records = []

    for agent in agents:
        p_col = f'{agent}_Profit'
        inv_col = f'{agent}_AvgInv'
        unf_col = f'{agent}_Unfulfilled'
        fr_col = f'{agent}_FillRate'

        if all(c in df.columns for c in [p_col, inv_col, unf_col, fr_col]):
            vals = df[df[p_col] > -5000] if p_col in df.columns else df
            records.append({
                'Agent': agent,
                'Profit': vals[p_col].mean(),
                'Avg Inventory': vals[inv_col].mean(),
                'Unfulfilled': vals[unf_col].mean(),
                'Fill Rate': vals[fr_col].mean(),
            })

    plot_df = pd.DataFrame(records)
    palette = get_palette(agents)

    fig, axes = plt.subplots(1, 3, figsize=FIG_WIDE)

    # Panel 1: Profit comparison
    ax1 = axes[0]
    bars = ax1.barh(
        plot_df['Agent'], plot_df['Profit'],
        color=[palette.get(a, '#666') for a in plot_df['Agent']],
        edgecolor='white', linewidth=0.3, height=0.6,
    )
    for bar, val in zip(bars, plot_df['Profit']):
        ax1.text(val - 20, bar.get_y() + bar.get_height() / 2,
                f'${val:.0f}', va='center', ha='right',
                fontsize=7, color='white', weight='bold')
    ax1.set_title('Profit ($)', fontsize=9, weight='bold')
    ax1.set_xlabel('')

    # Panel 2: Average Inventory (holding cost proxy)
    ax2 = axes[1]
    bars2 = ax2.barh(
        plot_df['Agent'], plot_df['Avg Inventory'],
        color=[palette.get(a, '#666') for a in plot_df['Agent']],
        edgecolor='white', linewidth=0.3, height=0.6,
    )
    for bar, val in zip(bars2, plot_df['Avg Inventory']):
        ax2.text(val + 5, bar.get_y() + bar.get_height() / 2,
                f'{val:.0f}', va='center', ha='left',
                fontsize=7, color='#333', weight='bold')
    ax2.set_title('Avg Inventory (units)', fontsize=9, weight='bold')
    ax2.set_yticklabels([])
    ax2.set_xlabel('')

    # Panel 3: Unfulfilled Demand (penalty proxy)
    ax3 = axes[2]
    bars3 = ax3.barh(
        plot_df['Agent'], plot_df['Unfulfilled'],
        color=[palette.get(a, '#666') for a in plot_df['Agent']],
        edgecolor='white', linewidth=0.3, height=0.6,
    )
    for bar, val in zip(bars3, plot_df['Unfulfilled']):
        ax3.text(val + 2, bar.get_y() + bar.get_height() / 2,
                f'{val:.0f}', va='center', ha='left',
                fontsize=7, color='#333', weight='bold')
    ax3.set_title('Unfulfilled Demand (units)', fontsize=9, weight='bold')
    ax3.set_yticklabels([])
    ax3.set_xlabel('')

    fig.suptitle('Operational Decomposition — How Agents Achieve Profit',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, 'E4_reward_decomposition'))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    input_csv = "data/results/benchmark_results_comprehensive_iterative.csv"
    out_dir = "data/results/charts_environment"
    os.makedirs(out_dir, exist_ok=True)

    print("E1  Demand Profile Gallery ...")
    plot_demand_gallery(out_dir)

    print("E2  Topology Comparison ...")
    plot_topology_comparison(out_dir)

    df = pd.read_csv(input_csv)

    print("E3  Complexity Landscape ...")
    plot_complexity_landscape(df, out_dir)

    print("E4  Reward Decomposition ...")
    plot_reward_decomposition(df, out_dir)

    print(f"\n✅ Environment charts saved to {out_dir}/")


if __name__ == "__main__":
    main()
