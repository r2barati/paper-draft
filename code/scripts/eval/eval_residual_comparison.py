import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
"""
eval_residual_comparison.py — Compare Residual RL V1 vs V2 vs Baselines

Evaluates the new Residual RL models (V1: flat MLP + DR, V2: shared MLP + DR)
across the benchmark scenario grid and compares against existing baseline results.
"""

import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.core.environment import CoreEnv
from src.envs.wrappers.feature_wrappers import DomainFeatureWrapper
from src.envs.wrappers.per_link_wrapper import PerLinkFeatureWrapper
from src.envs.wrappers.action_wrappers import ResidualActionWrapper
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.mssp_agent import RollingHorizonMSSPAgent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCENARIO = 'base'
DEMAND_TYPES = ['stationary', 'shock']
GOODWILL_FLAGS = [False, True]
NUM_SEEDS = 10
EVAL_SEEDS = list(range(100, 100 + NUM_SEEDS))
PLANNING_HORIZON = 30

# Model paths (from training)
V1_MODEL = 'data/models/ppo_residual_base.zip'
V1_STATS = 'data/models/vec_normalize_residual_base.pkl'

V2_MODEL = 'data/models/ppo_residual_v2_base.zip'
V2_STATS = 'data/models/vec_normalize_residual_v2_base.pkl'

OUTPUT_CSV = 'data/results/residual_comparison.csv'


# ---------------------------------------------------------------------------
# Evaluation Helpers
# ---------------------------------------------------------------------------

def evaluate_heuristic(env_kwargs, seed):
    """Run the Newsvendor heuristic baseline."""
    env = CoreEnv(**env_kwargs)
    obs, _ = env.reset(seed=seed)
    agent = HeuristicAgent(env, is_blind=False)

    total_reward = 0
    for t in range(PLANNING_HORIZON):
        action = agent.get_action(obs, t)
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        if term or trunc:
            break

    return {'profit': np.sum(env.P), 'reward': total_reward}


def evaluate_mssp(env_kwargs, seed):
    """Run the MSSP baseline."""
    env = CoreEnv(**env_kwargs)
    obs, _ = env.reset(seed=seed)
    agent = RollingHorizonMSSPAgent(env, planning_horizon=10)

    total_reward = 0
    for t in range(PLANNING_HORIZON):
        action = agent.get_action(t)
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        if term or trunc:
            break

    return {'profit': np.sum(env.P), 'reward': total_reward}


def evaluate_residual_v1(env_kwargs, seed, model, vec_normalize_env):
    """Evaluate Residual RL V1 (flat MLP + Domain Randomization)."""
    env = CoreEnv(**env_kwargs)
    env = DomainFeatureWrapper(env, is_blind=False)
    heuristic = HeuristicAgent(env.unwrapped, is_blind=False)
    env = ResidualActionWrapper(env, heuristic_agent=heuristic, max_residual=50.0)

    obs, _ = env.reset(seed=seed)

    # Normalize observation using training stats
    obs = vec_normalize_env.normalize_obs(obs)

    total_reward = 0
    for t in range(PLANNING_HORIZON):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        obs = vec_normalize_env.normalize_obs(obs)
        total_reward += reward
        if term or trunc:
            break

    return {'profit': np.sum(env.unwrapped.P), 'reward': total_reward}


def evaluate_residual_v2(env_kwargs, seed, model, vec_normalize_env):
    """Evaluate Residual RL V2 (shared MLP + Domain Randomization)."""
    env = CoreEnv(**env_kwargs)
    heuristic = HeuristicAgent(env, is_blind=False)
    env = PerLinkFeatureWrapper(env, heuristic_agent=heuristic, is_blind=False)
    env = ResidualActionWrapper(env, heuristic_agent=heuristic, max_residual=50.0)

    obs, _ = env.reset(seed=seed)

    # Normalize observation using training stats
    obs = vec_normalize_env.normalize_obs(obs)

    total_reward = 0
    for t in range(PLANNING_HORIZON):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        obs = vec_normalize_env.normalize_obs(obs)
        total_reward += reward
        if term or trunc:
            break

    return {'profit': np.sum(env.unwrapped.P), 'reward': total_reward}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Residual RL Comparison: V1 (Flat MLP) vs V2 (Shared MLP) vs Baselines")
    print("=" * 70)

    # Load trained models
    print("\nLoading V1 model...")
    v1_model = PPO.load(V1_MODEL)
    with open(V1_STATS, 'rb') as f:
        v1_vec = pickle.load(f)
    v1_vec.training = False
    v1_vec.norm_reward = False
    print(f"  V1 obs shape: {v1_model.observation_space.shape}")

    print("Loading V2 model...")
    v2_model = PPO.load(V2_MODEL)
    with open(V2_STATS, 'rb') as f:
        v2_vec = pickle.load(f)
    v2_vec.training = False
    v2_vec.norm_reward = False
    print(f"  V2 obs shape: {v2_model.observation_space.shape}")

    results = []

    for demand_type in DEMAND_TYPES:
        for use_goodwill in GOODWILL_FLAGS:
            desc = f"{demand_type} | GW={use_goodwill}"
            print(f"\n--- Scenario: {desc} ---")

            for seed in tqdm(EVAL_SEEDS, desc=desc):
                env_kwargs = {
                    'scenario': SCENARIO,
                    'num_periods': PLANNING_HORIZON,
                    'demand_config': {
                        'type': demand_type,
                        'base_mu': 20,
                        'use_goodwill': use_goodwill,
                    }
                }

                # Heuristic baseline
                try:
                    heur = evaluate_heuristic(env_kwargs, seed)
                    results.append({
                        'agent': 'Heuristic', 'demand': demand_type,
                        'goodwill': use_goodwill, 'seed': seed,
                        'profit': heur['profit']
                    })
                except Exception as e:
                    print(f"  Heuristic error: {e}")

                # MSSP baseline
                try:
                    mssp = evaluate_mssp(env_kwargs, seed)
                    results.append({
                        'agent': 'MSSP', 'demand': demand_type,
                        'goodwill': use_goodwill, 'seed': seed,
                        'profit': mssp['profit']
                    })
                except Exception as e:
                    print(f"  MSSP error: {e}")

                # Residual V1
                try:
                    v1 = evaluate_residual_v1(env_kwargs, seed, v1_model, v1_vec)
                    results.append({
                        'agent': 'Residual_V1', 'demand': demand_type,
                        'goodwill': use_goodwill, 'seed': seed,
                        'profit': v1['profit']
                    })
                except Exception as e:
                    print(f"  V1 error: {e}")

                # Residual V2
                try:
                    v2 = evaluate_residual_v2(env_kwargs, seed, v2_model, v2_vec)
                    results.append({
                        'agent': 'Residual_V2', 'demand': demand_type,
                        'goodwill': use_goodwill, 'seed': seed,
                        'profit': v2['profit']
                    })
                except Exception as e:
                    print(f"  V2 error: {e}")

    # --- Results ---
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
    print(pivot.to_string())

    # Save
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nRaw results saved to {OUTPUT_CSV}")

    # --- Improvement Analysis ---
    print("\n" + "=" * 70)
    print("IMPROVEMENT OVER HEURISTIC (% profit gain)")
    print("=" * 70)

    for demand_type in DEMAND_TYPES:
        for use_goodwill in GOODWILL_FLAGS:
            mask = (df['demand'] == demand_type) & (df['goodwill'] == use_goodwill)
            sub = df[mask]

            heur_mean = sub[sub['agent'] == 'Heuristic']['profit'].mean()
            if heur_mean == 0:
                continue

            for agent in ['MSSP', 'Residual_V1', 'Residual_V2']:
                agent_mean = sub[sub['agent'] == agent]['profit'].mean()
                pct = (agent_mean - heur_mean) / abs(heur_mean) * 100
                print(f"  {demand_type:12s} | GW={str(use_goodwill):5s} | {agent:15s}: {agent_mean:8.2f}  ({pct:+.1f}%)")


if __name__ == '__main__':
    main()
