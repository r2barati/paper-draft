"""
config.py — Centralised scenario matrix and tier definitions.

To add a new demand scenario, simply append it to DEMAND_TYPES.
To add a new RL model, append a dict to RL_MODELS in runners.py.
"""

# ---------------------------------------------------------------------------
# Scenario Matrix Dimensions
# ---------------------------------------------------------------------------
NETWORKS = ['base', 'serial']
DEMAND_TYPES = [
    'stationary', 'trend', 'seasonal', 'shock',
    'trend+seasonal', 'trend+shock', 'seasonal+shock', 'trend+seasonal+shock',
]
GOODWILL_FLAGS = [False, True]
BACKLOG_FLAGS = [True, False]
PLANNING_HORIZON = 30

# ---------------------------------------------------------------------------
# Demand Config Mapping
# Maps a demand_type string to the dict passed to demand_config.
# Single-effect types use the legacy 'type' key for backward compat.
# Composable types use the new 'effects' list API.
# ---------------------------------------------------------------------------
DEMAND_CONFIGS = {
    'stationary':             {'type': 'stationary',  'base_mu': 20},
    'trend':                  {'type': 'trend',       'base_mu': 20},
    'seasonal':               {'type': 'seasonal',    'base_mu': 20},
    'shock':                  {'type': 'shock',       'base_mu': 20},
    'trend+seasonal':         {'effects': ['trend', 'seasonal'],        'base_mu': 20},
    'trend+shock':            {'effects': ['trend', 'shock'],           'base_mu': 20},
    'seasonal+shock':         {'effects': ['seasonal', 'shock'],        'base_mu': 20},
    'trend+seasonal+shock':   {'effects': ['trend', 'seasonal', 'shock'], 'base_mu': 20},
}

# ---------------------------------------------------------------------------
# Tier Configurations
# ---------------------------------------------------------------------------
TIER_CONFIG = {
    1: {  # Smoke Test (~1-2 min)
        'description': 'Smoke test — validates all agents run without crashing',
        'networks': ['base'],            # use base — RL models are trained on base
        'demand_types': ['shock'],       # single hardest scenario
        'goodwill': [False],
        'backlog': [True],
        'seeds': [100],                  # single seed
    },
    2: {  # Sanity Check (~10-15 min)
        'description': 'Sanity check — validates numbers are in the right ballpark',
        'networks': ['base'],
        'demand_types': DEMAND_TYPES,    # all demand types
        'goodwill': GOODWILL_FLAGS,
        'backlog': [True],
        'seeds': list(range(100, 103)),  # 3 seeds
    },
    3: {  # Comprehensive (hours)
        'description': 'Comprehensive — full scenario matrix for publication-quality results',
        'networks': NETWORKS,
        'demand_types': DEMAND_TYPES,
        'goodwill': GOODWILL_FLAGS,
        'backlog': BACKLOG_FLAGS,
        'seeds': None,                   # set via --seeds CLI arg (default 5)
    },
}

# ---------------------------------------------------------------------------
# Agent Execution Order (cheap → expensive for fail-fast behaviour)
# ---------------------------------------------------------------------------
AGENT_ORDER = [
    'Dummy',
    'Newsvendor', 'Newsvendor_Blind',
    'SS_Policy', 'SS_Policy_Blind',
    'ExpSmoothing', 'ExpSmoothing_Blind',
    # RL models (cheap inference)
    'RLV4', 'RLGNN', 'Residual',
    # OR solvers (expensive)
    'DLP', 'DLP_Blind',
    'MSSP', 'MSSP_Blind',
    'Oracle',
]

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
RESULTS_DIR = 'data/results'
REFERENCE_FILE = 'data/results/benchmark_reference.json'
CACHE_DIR = 'data/results'
LOG_DIR = './data/logs/benchmark'
