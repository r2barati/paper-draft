"""
runners.py — Agent execution functions for the benchmark.

All runners follow the interface:
    runner(env_kwargs, seed, planning_horizon, **kwargs) → dict | None

Each runner uses make_supply_chain_env() from the unified builder.
"""

import os
import numpy as np
import pickle

from src.envs.builder import make_supply_chain_env
from src.agents.oracle import StandaloneOracleOptimizer
from src.agents.dlp_agent import RollingHorizonDLPAgent
from src.agents.mssp_agent import RollingHorizonMSSPAgent
from src.agents.heuristic_agent import HeuristicAgent

from .config import LOG_DIR


def compute_adaptive_horizon(env_kwargs, buffer=10):
    """Compute agent planning horizon = max_cumulative_pipeline_LT + buffer.

    Uses BFS through the network's reorder links to find the longest
    cumulative lead-time path from raw materials to retail.
    """
    from collections import deque
    from src.envs.core.network_topology import SupplyChainNetwork

    net = SupplyChainNetwork(scenario=env_kwargs.get('scenario', 'base'))
    num_periods = env_kwargs.get('num_periods', 30)

    max_cumulative_lt = 0
    for rm in net.rawmat:
        queue = deque([(rm, 0)])
        visited = set()
        while queue:
            node, cum_lt = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            max_cumulative_lt = max(max_cumulative_lt, cum_lt)
            for (u, v) in net.reorder_links:
                if u == node:
                    lt = net.graph.edges[(u, v)].get('L', 0)
                    queue.append((v, cum_lt + lt))

    return min(max_cumulative_lt + buffer, num_periods)

# ---------------------------------------------------------------------------
# KPI extraction (shared across all runners)
# ---------------------------------------------------------------------------
def compute_kpi(env, episode_reward):
    """Extract standard KPIs from a completed episode."""
    base_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    avg_inv = np.mean([np.sum(base_env.X[t]) for t in range(base_env.period)]) if base_env.period > 0 else 0
    total_u = np.sum(base_env.U)
    total_d = np.sum(base_env.D)
    fill_rate = max(0.0, 1.0 - (total_u / total_d)) if total_d > 0 else 1.0
    return {
        'profit': np.sum(base_env.P),
        'avg_inv': avg_inv,
        'unfulfilled': total_u,
        'fill_rate': fill_rate,
        'avg_backlog': np.mean(np.sum(base_env.U, axis=1)),
        'final_sentiment': base_env.demand_engine.sentiment,
    }


# ===========================================================================
# OR Baseline Runners
# ===========================================================================

def run_oracle(env_kwargs, seed, planning_horizon, **kwargs):
    """Oracle: perfect-information LP upper bound."""
    use_goodwill = env_kwargs.get('demand_config', {}).get('use_goodwill', False)

    if use_goodwill:
        from src.agents.baselines import OptimisticEndogenousOracle
        # Build a raw env for the oracle to read topology
        _raw_kwargs = {k: v for k, v in env_kwargs.items()}
        oracle = OptimisticEndogenousOracle(_raw_kwargs, planning_horizon=planning_horizon)
        optimal_actions = oracle.solve(seed)
        if optimal_actions is None:
            return None

        env = make_supply_chain_env(agent_type='or', use_integer_actions=False, **env_kwargs)
        env.reset(seed=seed)
        episode_reward = 0
        for t in range(planning_horizon):
            _, reward, _, _, _ = env.step(optimal_actions[t])
            episode_reward += reward
        return compute_kpi(env, episode_reward)
    else:
        # Probe run to extract realised demand, then solve LP on fixed demand
        from src.envs.core.environment import CoreEnv
        probe_env = CoreEnv(**env_kwargs)
        probe_env.reset(seed=seed)
        for _ in range(planning_horizon):
            probe_env.step(np.zeros(probe_env.action_space.shape))
        demand_trace = {}
        for i, edge in enumerate(probe_env.network.retail_links):
            demand_trace[edge] = probe_env.D[:, i].copy()

        fixed_cfg = env_kwargs.copy()
        fixed_cfg['user_D'] = demand_trace
        env = CoreEnv(**fixed_cfg)
        env.reset(seed=seed)
        oracle = StandaloneOracleOptimizer(
            env, known_demand_scenario=demand_trace,
            planning_horizon=planning_horizon, is_continuous=True)
        optimal_actions = oracle.solve_full_horizon()
        if optimal_actions is None:
            return None
        episode_reward = 0
        for t in range(planning_horizon):
            _, reward, _, _, _ = env.step(optimal_actions[t])
            episode_reward += reward
        return compute_kpi(env, episode_reward)


def run_dlp(env_kwargs, seed, planning_horizon, is_blind=False, agent_horizon=None, **kwargs):
    """Rolling Horizon DLP Agent.

    Args:
        agent_horizon: LP look-ahead window per step. If None, uses
                       adaptive horizon = max_pipeline_LT + 10.
                       Automatically shrinks to remaining periods at end of episode.
    """
    env = make_supply_chain_env(agent_type='or', use_integer_actions=True, **env_kwargs)
    obs, _ = env.reset(seed=seed)

    if agent_horizon is None:
        agent_horizon = compute_adaptive_horizon(env_kwargs)

    agent = RollingHorizonDLPAgent(env.unwrapped, planning_horizon=agent_horizon, is_blind=is_blind)
    episode_reward = 0
    for t in range(planning_horizon):
        # Shrink horizon at end of episode so LP doesn't plan past the end
        remaining = planning_horizon - t
        if remaining < agent_horizon:
            agent.H = remaining
        obs, reward, terminated, truncated, _ = env.step(agent.get_action(t))
        episode_reward += reward
        if terminated or truncated:
            break
    return compute_kpi(env, episode_reward)


def run_mssp(env_kwargs, seed, planning_horizon, is_blind=False, agent_horizon=None, **kwargs):
    """Rolling Horizon MSSP Agent (MIP solver)."""
    env = make_supply_chain_env(agent_type='or', use_integer_actions=True, **env_kwargs)
    obs, _ = env.reset(seed=seed)

    if agent_horizon is None:
        agent_horizon = min(10, planning_horizon)  # MSSP is expensive, keep H=10

    agent = RollingHorizonMSSPAgent(env.unwrapped, planning_horizon=agent_horizon, branching_depth=3, is_blind=is_blind)
    episode_reward = 0
    for t in range(planning_horizon):
        obs, reward, terminated, truncated, _ = env.step(agent.get_action(t))
        episode_reward += reward
        if terminated or truncated:
            break
    return compute_kpi(env, episode_reward)


def run_heuristic(env_kwargs, seed, planning_horizon, is_blind=False, **kwargs):
    """Base-Stock Heuristic Agent."""
    env = make_supply_chain_env(agent_type='or', use_integer_actions=True, **env_kwargs)
    obs, _ = env.reset(seed=seed)
    agent = HeuristicAgent(env.unwrapped, is_blind=is_blind)
    episode_reward = 0
    for t in range(planning_horizon):
        obs, reward, terminated, truncated, _ = env.step(agent.get_action(obs, t))
        episode_reward += reward
        if terminated or truncated:
            break
    return compute_kpi(env, episode_reward)


def run_dummy(env_kwargs, seed, planning_horizon, **kwargs):
    """Random action agent — lower bound."""
    env = make_supply_chain_env(agent_type='or', use_integer_actions=True, **env_kwargs)
    obs, _ = env.reset(seed=seed)
    episode_reward = 0
    for _ in range(planning_horizon):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        episode_reward += reward
        if terminated or truncated:
            break
    return compute_kpi(env, episode_reward)


# ===========================================================================
# RL Runners
# ===========================================================================

def _load_ppo_and_normalizer(model_path, stats_path, custom_objects=None):
    """Shared helper: load PPO model + VecNormalize stats."""
    from stable_baselines3 import PPO

    zip_path = model_path if model_path.endswith('.zip') else model_path + '.zip'
    if not os.path.exists(zip_path):
        return None, None

    model = PPO.load(model_path, custom_objects=custom_objects)

    with open(stats_path, 'rb') as f:
        norm_env = pickle.load(f)
    norm_env.training = False
    norm_env.norm_reward = False

    return model, norm_env


def run_rl_standard(env_kwargs, seed, planning_horizon,
                    model_path=None, stats_path=None, **kwargs):
    """Standard RL agent (V1–V4). Uses unified builder with agent_type='rl'."""
    model, norm_env = _load_ppo_and_normalizer(model_path, stats_path)
    if model is None:
        return None

    env = make_supply_chain_env(agent_type='rl', use_integer_actions=True, **env_kwargs)
    obs, _ = env.reset(seed=seed)

    # Check obs-shape compatibility: model expects a certain obs size from training
    expected_shape = norm_env.obs_rms.mean.shape[0]
    if len(obs) < expected_shape:
        # Model was trained on a larger network — incompatible
        return None

    episode_reward = 0
    for t in range(planning_horizon):
        trimmed_obs = obs[-expected_shape:]
        obs_2d = trimmed_obs.reshape(1, -1)
        norm_obs = norm_env.normalize_obs(obs_2d).squeeze(0)
        action, _ = model.predict(norm_obs, deterministic=True)

        # Clip action to current network size (handle base→serial mismatch)
        action = action[:len(env.unwrapped.network.reorder_links)]

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += info.get('raw_reward', reward)
        if terminated or truncated:
            break

    return compute_kpi(env, episode_reward)


def run_rl_gnn(env_kwargs, seed, planning_horizon,
               model_path=None, stats_path=None, **kwargs):
    """GNN-based RL agent. Uses agent_type='gnn' and GNNFeaturesExtractor."""
    from src.models.gnn_extractor import GNNFeaturesExtractor

    custom_objects = {"GNNFeaturesExtractor": GNNFeaturesExtractor}
    model, norm_env = _load_ppo_and_normalizer(model_path, stats_path, custom_objects)
    if model is None:
        return None

    env = make_supply_chain_env(agent_type='gnn', use_integer_actions=True, **env_kwargs)
    obs, _ = env.reset(seed=seed)

    # Check obs-shape compatibility
    expected_shape = norm_env.obs_rms.mean.shape[0]
    if len(obs) < expected_shape:
        return None

    episode_reward = 0
    for t in range(planning_horizon):
        trimmed_obs = obs[-expected_shape:]
        obs_2d = trimmed_obs.reshape(1, -1)
        norm_obs = norm_env.normalize_obs(obs_2d).squeeze(0)
        action, _ = model.predict(norm_obs, deterministic=True)

        action = action[:len(env.unwrapped.network.reorder_links)]

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += info.get('raw_reward', reward)
        if terminated or truncated:
            break

    return compute_kpi(env, episode_reward)


def run_rl_residual(env_kwargs, seed, planning_horizon,
                    model_path=None, stats_path=None, **kwargs):
    """Residual RL agent. Builds heuristic internally, uses agent_type='residual_rl'."""
    model, norm_env = _load_ppo_and_normalizer(model_path, stats_path)
    if model is None:
        return None

    # Build headless heuristic for the residual wrapper
    dummy_env = make_supply_chain_env(agent_type='or', **env_kwargs)
    heuristic = HeuristicAgent(dummy_env.unwrapped, is_blind=False)

    env = make_supply_chain_env(
        agent_type='residual_rl', heuristic_agent=heuristic,
        max_residual=50.0, **env_kwargs
    )
    obs, _ = env.reset(seed=seed)
    episode_reward = 0

    for t in range(planning_horizon):
        obs_2d = obs.reshape(1, -1)
        norm_obs = norm_env.normalize_obs(obs_2d).squeeze(0)
        action, _ = model.predict(norm_obs, deterministic=True)

        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        if terminated or truncated:
            break

    return compute_kpi(env, episode_reward)


# ===========================================================================
# Agent Registry
# ===========================================================================

AGENT_REGISTRY = {
    # --- OR Baselines ---
    'Oracle':          {'runner': run_oracle, 'category': 'or'},
    'MSSP':            {'runner': run_mssp,   'category': 'or', 'kwargs': {'is_blind': False}},
    'MSSP_Blind':      {'runner': run_mssp,   'category': 'or', 'kwargs': {'is_blind': True}},
    'DLP':             {'runner': run_dlp,    'category': 'or', 'kwargs': {'is_blind': False}},
    'DLP_Blind':       {'runner': run_dlp,    'category': 'or', 'kwargs': {'is_blind': True}},
    'Heuristic':       {'runner': run_heuristic, 'category': 'or', 'kwargs': {'is_blind': False}},
    'Heuristic_Blind': {'runner': run_heuristic, 'category': 'or', 'kwargs': {'is_blind': True}},
    'Dummy':           {'runner': run_dummy,  'category': 'or'},

    # --- RL Models ---
    'RLGNN':    {'runner': run_rl_gnn,      'category': 'rl',
                 'kwargs': {'model_path': 'data/models/ppo_gnn.zip',
                            'stats_path': 'data/models/vec_normalize_gnn.pkl'}},
    'Residual': {'runner': run_rl_residual, 'category': 'rl',
                 'kwargs': {'model_path': 'data/models/ppo_residual.zip',
                            'stats_path': 'data/models/vec_normalize_residual.pkl'}},
}
