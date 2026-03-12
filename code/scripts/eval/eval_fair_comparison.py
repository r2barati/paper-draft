import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
"""
eval_fair_comparison.py — Comprehensive Fair Comparison

Compares scenario-specific Residual V2 models vs the general (DR-trained)
model, Heuristic, MSSP, DLP, and Oracle across all scenario combinations.

Each scenario-specific RL model is evaluated ONLY on the scenario it was
trained on — matching the information parity with MSSP/DLP.
"""

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from stable_baselines3 import PPO

from src.envs.core.environment import CoreEnv
from src.envs.wrappers.per_link_wrapper import PerLinkFeatureWrapper
from src.envs.wrappers.feature_wrappers import DomainFeatureWrapper
from src.envs.wrappers.action_wrappers import ProportionalResidualWrapper
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.mssp_agent import RollingHorizonMSSPAgent
from src.agents.dlp_agent import RollingHorizonDLPAgent
from src.agents.baselines import OptimisticEndogenousOracle

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCENARIO = 'base'
DEMAND_TYPES = ['stationary', 'shock']
GOODWILL_FLAGS = [False, True]
NUM_SEEDS = 10
EVAL_SEEDS = list(range(100, 100 + NUM_SEEDS))
PLANNING_HORIZON = 30

# Paths
GENERAL_MODEL = 'data/models/ppo_residual_v2_base.zip'
GENERAL_STATS = 'data/models/vec_normalize_residual_v2_base.pkl'
OUTPUT_CSV = 'data/results/fair_comparison.csv'


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def _make_env(demand_type, use_goodwill, seed):
    return dict(
        scenario=SCENARIO,
        num_periods=PLANNING_HORIZON,
        demand_config={'type': demand_type, 'base_mu': 20, 'use_goodwill': use_goodwill},
    )


def evaluate_heuristic(env_kwargs, seed):
    env = CoreEnv(**env_kwargs)
    obs, _ = env.reset(seed=seed)
    agent = HeuristicAgent(env, is_blind=False)
    total_reward = 0
    for t in range(PLANNING_HORIZON):
        action = agent.get_action(obs, t)
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        if term or trunc: break
    return np.sum(env.P)


def evaluate_mssp(env_kwargs, seed):
    env = CoreEnv(**env_kwargs)
    obs, _ = env.reset(seed=seed)
    agent = RollingHorizonMSSPAgent(env, planning_horizon=10)
    total_reward = 0
    for t in range(PLANNING_HORIZON):
        action = agent.get_action(t)
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        if term or trunc: break
    return np.sum(env.P)


def evaluate_dlp(env_kwargs, seed):
    env = CoreEnv(**env_kwargs)
    obs, _ = env.reset(seed=seed)
    agent = RollingHorizonDLPAgent(env, planning_horizon=10)
    total_reward = 0
    for t in range(PLANNING_HORIZON):
        action = agent.get_action(t)
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        if term or trunc: break
    return np.sum(env.P)


def evaluate_mssp_blind(env_kwargs, seed):
    env = CoreEnv(**env_kwargs)
    obs, _ = env.reset(seed=seed)
    agent = RollingHorizonMSSPAgent(env, planning_horizon=10, is_blind=True)
    total_reward = 0
    for t in range(PLANNING_HORIZON):
        action = agent.get_action(t)
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        if term or trunc: break
    return np.sum(env.P)


def evaluate_dlp_blind(env_kwargs, seed):
    env = CoreEnv(**env_kwargs)
    obs, _ = env.reset(seed=seed)
    agent = RollingHorizonDLPAgent(env, planning_horizon=10, is_blind=True)
    total_reward = 0
    for t in range(PLANNING_HORIZON):
        action = agent.get_action(t)
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        if term or trunc: break
    return np.sum(env.P)


def evaluate_oracle(env_kwargs, seed):
    optimizer = OptimisticEndogenousOracle(env_kwargs, PLANNING_HORIZON)
    action_matrix = optimizer.solve(seed=seed)
    # Now run the actual environment with the oracle's actions
    env = CoreEnv(**env_kwargs)
    obs, _ = env.reset(seed=seed)
    total_reward = 0
    for t in range(PLANNING_HORIZON):
        obs, reward, term, trunc, _ = env.step(action_matrix[t])
        total_reward += reward
        if term or trunc: break
    return np.sum(env.P)


def evaluate_residual_v2(env_kwargs, seed, model, vec_normalize):
    """Evaluate Residual RL V2 (shared MLP)."""
    env = CoreEnv(**env_kwargs)
    heuristic = HeuristicAgent(env, is_blind=False)
    env = PerLinkFeatureWrapper(env, heuristic_agent=heuristic, is_blind=False)
    env = ProportionalResidualWrapper(env, heuristic_agent=heuristic, max_pct=0.5, reward_lambda=0.0)
    obs, _ = env.reset(seed=seed)
    obs = vec_normalize.normalize_obs(obs)
    total_reward = 0
    for t in range(PLANNING_HORIZON):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        obs = vec_normalize.normalize_obs(obs)
        total_reward += reward
        if term or trunc: break
    return np.sum(env.unwrapped.P)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("FAIR COMPARISON: Scenario-Specific vs General Residual RL vs Baselines")
    print("=" * 70)

    # Load general (DR) model
    print("\nLoading General (DR) Residual V2 model...")
    gen_model = PPO.load(GENERAL_MODEL)
    with open(GENERAL_STATS, 'rb') as f:
        gen_vec = pickle.load(f)
    gen_vec.training = False
    gen_vec.norm_reward = False

    # Load scenario-specific models
    specific_models = {}
    for dt in DEMAND_TYPES:
        for gw in GOODWILL_FLAGS:
            gw_tag = "gw" if gw else "nogw"
            tag = f"{dt}_{gw_tag}"
            model_path = f'data/models/ppo_residual_v2_specific_{tag}.zip'
            stats_path = f'data/models/vec_normalize_residual_v2_specific_{tag}.pkl'
            if os.path.exists(model_path):
                m = PPO.load(model_path)
                with open(stats_path, 'rb') as f:
                    v = pickle.load(f)
                v.training = False
                v.norm_reward = False
                specific_models[tag] = (m, v)
                print(f"  Loaded specific model: {tag}")
            else:
                print(f"  [MISSING] {model_path}")

    results = []

    for demand_type in DEMAND_TYPES:
        for use_goodwill in GOODWILL_FLAGS:
            gw_tag = "gw" if use_goodwill else "nogw"
            tag = f"{demand_type}_{gw_tag}"
            desc = f"{demand_type} | GW={use_goodwill}"
            print(f"\n--- Scenario: {desc} ---")

            for seed in tqdm(EVAL_SEEDS, desc=desc):
                env_kwargs = _make_env(demand_type, use_goodwill, seed)

                # --- Oracle ---
                try:
                    profit = evaluate_oracle(env_kwargs, seed)
                    results.append({
                        'agent': 'Oracle', 'demand': demand_type,
                        'goodwill': use_goodwill, 'seed': seed, 'profit': profit
                    })
                except Exception as e:
                    print(f"  Oracle error: {e}")

                # --- Heuristic ---
                try:
                    profit = evaluate_heuristic(env_kwargs, seed)
                    results.append({
                        'agent': 'Heuristic', 'demand': demand_type,
                        'goodwill': use_goodwill, 'seed': seed, 'profit': profit
                    })
                except Exception as e:
                    print(f"  Heuristic error: {e}")

                # --- DLP ---
                try:
                    profit = evaluate_dlp(env_kwargs, seed)
                    results.append({
                        'agent': 'DLP', 'demand': demand_type,
                        'goodwill': use_goodwill, 'seed': seed, 'profit': profit
                    })
                except Exception as e:
                    print(f"  DLP error: {e}")

                # --- MSSP ---
                try:
                    profit = evaluate_mssp(env_kwargs, seed)
                    results.append({
                        'agent': 'MSSP', 'demand': demand_type,
                        'goodwill': use_goodwill, 'seed': seed, 'profit': profit
                    })
                except Exception as e:
                    print(f"  MSSP error: {e}")

                # --- MSSP Blind ---
                try:
                    profit = evaluate_mssp_blind(env_kwargs, seed)
                    results.append({
                        'agent': 'MSSP_Blind', 'demand': demand_type,
                        'goodwill': use_goodwill, 'seed': seed, 'profit': profit
                    })
                except Exception as e:
                    print(f"  MSSP_Blind error: {e}")

                # --- DLP Blind ---
                try:
                    profit = evaluate_dlp_blind(env_kwargs, seed)
                    results.append({
                        'agent': 'DLP_Blind', 'demand': demand_type,
                        'goodwill': use_goodwill, 'seed': seed, 'profit': profit
                    })
                except Exception as e:
                    print(f"  DLP_Blind error: {e}")

                # --- Residual V2 General (DR) ---
                try:
                    profit = evaluate_residual_v2(env_kwargs, seed, gen_model, gen_vec)
                    results.append({
                        'agent': 'Residual_General', 'demand': demand_type,
                        'goodwill': use_goodwill, 'seed': seed, 'profit': profit
                    })
                except Exception as e:
                    print(f"  Residual_General error: {e}")

                # --- Residual V2 Scenario-Specific ---
                if tag in specific_models:
                    try:
                        spec_model, spec_vec = specific_models[tag]
                        profit = evaluate_residual_v2(env_kwargs, seed, spec_model, spec_vec)
                        results.append({
                            'agent': 'Residual_Specific', 'demand': demand_type,
                            'goodwill': use_goodwill, 'seed': seed, 'profit': profit
                        })
                    except Exception as e:
                        print(f"  Residual_Specific error: {e}")

    # --- Output ---
    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("RESULTS: Mean Profit by Agent × Scenario")
    print("=" * 70)

    pivot = df.pivot_table(
        index='agent',
        columns=['demand', 'goodwill'],
        values='profit',
        aggfunc='mean'
    ).round(2)

    # Reorder agents for readability
    agent_order = ['Oracle', 'MSSP', 'MSSP_Blind', 'DLP', 'DLP_Blind', 'Residual_Specific', 'Residual_General', 'Heuristic']
    pivot = pivot.reindex([a for a in agent_order if a in pivot.index])
    print(pivot.to_string())

    # Save CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nRaw results saved to {OUTPUT_CSV}")

    # --- Gap to Oracle ---
    print("\n" + "=" * 70)
    print("OPTIMALITY GAP (% below Oracle)")
    print("=" * 70)

    for demand_type in DEMAND_TYPES:
        for use_goodwill in GOODWILL_FLAGS:
            mask = (df['demand'] == demand_type) & (df['goodwill'] == use_goodwill)
            sub = df[mask]
            oracle_mean = sub[sub['agent'] == 'Oracle']['profit'].mean()
            if oracle_mean == 0:
                continue
            print(f"\n  {demand_type} | GW={use_goodwill}  (Oracle: {oracle_mean:.2f})")
            for agent in ['MSSP', 'MSSP_Blind', 'DLP', 'DLP_Blind', 'Residual_Specific', 'Residual_General', 'Heuristic']:
                agent_mean = sub[sub['agent'] == agent]['profit'].mean()
                if agent_mean:
                    gap = (oracle_mean - agent_mean) / abs(oracle_mean) * 100
                    print(f"    {agent:20s}: {agent_mean:8.2f}  (gap: {gap:+.1f}%)")


if __name__ == '__main__':
    main()
