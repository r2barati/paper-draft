"""
chart_style.py — Shared publication-quality defaults for all chart scripts.

Import this module at the top of every chart script:
    from chart_style import apply_style, AGENT_PALETTE, AGENT_ORDER, save_fig, FIG_SINGLE, FIG_DOUBLE
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os

# ── Figure size presets (inches) ──────────────────────────────────────────────
FIG_SINGLE = (4.5, 3.5)    # single-column (journals)
FIG_DOUBLE = (7.5, 4.5)    # double-column / full-width
FIG_WIDE   = (9, 5)        # presentations / wide charts

# ── Consistent agent colour palette (colorblind-safe) ─────────────────────────
AGENT_PALETTE = {
    'Oracle':         '#2c7bb6',   # steel blue
    'MSSP':           '#d7191c',   # red
    'MSSP (Blind)':   '#f4a582',   # light red
    'DLP':            '#e6550d',   # orange
    'DLP (Blind)':    '#fdae61',   # light orange
    'Heuristic':      '#756bb1',   # purple
    'Heuristic (Blind)': '#bcbddc',# light purple
    'Dummy':          '#969696',   # gray
    'RLV1':           '#bdbdbd',   # light gray
    'RLV4':           '#abd9e9',   # light blue
    'Residual':       '#fdae61',   # amber
    'RLGNN':          '#1a9641',   # green
}

# ── Canonical display order ───────────────────────────────────────────────────
AGENT_ORDER = [
    'Oracle', 'RLGNN', 'MSSP', 'Residual', 'RLV4', 'RLV1',
    'DLP', 'Heuristic', 'Dummy',
    'MSSP (Blind)', 'DLP (Blind)', 'Heuristic (Blind)',
]

# ── Demand ordering ───────────────────────────────────────────────────────────
DEMAND_ORDER = ['stationary', 'trend', 'seasonal', 'shock']

# ── Human-readable label maps ─────────────────────────────────────────────────
BLIND_LABELS = {True: 'Partial Observability', False: 'Full Visibility'}
GOODWILL_LABELS = {True: 'With Goodwill', False: 'No Goodwill'}
BACKLOG_LABELS = {True: 'With Backlog', False: 'No Backlog (Lost Sales)'}


def apply_style():
    """Apply publication-quality matplotlib/seaborn defaults. Call once at module load."""
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.15)

    mpl.rcParams.update({
        # Fonts
        'font.family':       'serif',
        'font.serif':        ['Times New Roman', 'DejaVu Serif', 'Bitstream Vera Serif'],
        'mathtext.fontset':  'stix',
        'font.size':         9,
        'axes.titlesize':    11,
        'axes.labelsize':    10,
        'xtick.labelsize':   8,
        'ytick.labelsize':   8,
        'legend.fontsize':   8,

        # Lines & markers
        'lines.linewidth':   1.5,
        'lines.markersize':  5,

        # Grid
        'axes.grid':         True,
        'grid.alpha':        0.35,
        'grid.linewidth':    0.5,

        # Borders
        'axes.linewidth':    0.6,
        'axes.edgecolor':    '#333333',

        # Tight
        'figure.autolayout': True,
        'savefig.bbox':      'tight',
        'savefig.pad_inches': 0.05,
        'savefig.dpi':       300,
    })


def get_palette(agents=None):
    """Return palette list or dict filtered to the given agent names."""
    if agents is None:
        return AGENT_PALETTE
    return {a: AGENT_PALETTE.get(a, '#333333') for a in agents}


def get_agent_order(agents):
    """Return agents sorted in canonical order, keeping only those present."""
    return [a for a in AGENT_ORDER if a in agents]


def save_fig(fig, path_without_ext):
    """Save figure as both high-res PNG and vector PDF."""
    os.makedirs(os.path.dirname(path_without_ext), exist_ok=True)
    fig.savefig(f"{path_without_ext}.png", dpi=300)
    fig.savefig(f"{path_without_ext}.pdf")
    plt.close(fig)
    print(f"  → saved {path_without_ext}.png  +  .pdf")


def add_bar_labels(ax, fmt='{:.0f}', fontsize=6, color='#333', offset=0,
                   inside=False):
    """Add value labels to every bar in an axes."""
    for p in ax.patches:
        width = p.get_width() if p.get_width() != 0 else p.get_height()
        height = p.get_height()
        # Horizontal bars
        if abs(p.get_width()) > abs(p.get_height()):
            x = p.get_width()
            y = p.get_y() + p.get_height() / 2
            ha = 'right' if inside else 'left'
            va = 'center'
            ax.text(x + offset, y, fmt.format(x), ha=ha, va=va,
                    fontsize=fontsize, color=color, weight='semibold')
        else:
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ha = 'center'
            va = 'bottom'
            ax.text(x, y + offset, fmt.format(y), ha=ha, va=va,
                    fontsize=fontsize, color=color, weight='semibold')
