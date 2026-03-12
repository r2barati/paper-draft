import os
import argparse
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.envs.builder import make_supply_chain_env
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.mssp_agent import RollingHorizonMSSPAgent

# Set small configurations for the lightweight test
SCENARIO = 'base'
DEMAND_TYPES = ['shock']
GOODWILL_FLAGS = [False]
NUM_SEEDS = 3
EVAL_SEEDS = list(range(100, 100 + NUM_SEEDS))
PLANNING_HORIZON = 30
LOG_DIR = "./data/logs/lightweight_benchmark_test"

def _kpi(env, episode_reward):
    """Extract standard evaluation KPIs from the unwrapped env."""
    sum_profit = np.sum(env.P)
    avg_inventory_per_node = np.mean(env.X, axis=0)
    avg_backlog = np.mean(np.sum(env.U, axis=1))
    
    return {
        'profit': sum_profit,
        'avg_backlog': avg_backlog,
        'final_sentiment': env.demand_engine.sentiment
    }

def run_heuristic(env_kwargs, seed):
    """Run the Base-Stock Heuristic Agent."""
    run_name = f"Heuristic_{env_kwargs['scenario']}_{env_kwargs['demand_config']['type']}_gw{env_kwargs['demand_config']['use_goodwill']}_seed{seed}"
    
    env = make_supply_chain_env(
        agent_type='or',
        use_integer_actions=True,
        enable_logging=True,
        run_name=run_name,
        log_dir=LOG_DIR,
        record_trajectory=True, # Will save .npz trajectories
        **env_kwargs
    )
    
    obs, _ = env.reset(seed=seed)
    agent = HeuristicAgent(env.unwrapped, is_blind=False)
    
    episode_reward = 0
    for t in range(PLANNING_HORIZON):
        action = agent.get_action(obs, t)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        if terminated or truncated:
            break
            
    metrics = _kpi(env.unwrapped, episode_reward)
    env.close() # Triggers parquet extraction from Logging Wrapper
    return metrics

def run_mssp(env_kwargs, seed):
    """Run the Mathematical Programming (MSSP) Agent."""
    run_name = f"MSSP_{env_kwargs['scenario']}_{env_kwargs['demand_config']['type']}_gw{env_kwargs['demand_config']['use_goodwill']}_seed{seed}"
    
    env = make_supply_chain_env(
        agent_type='or',
        use_integer_actions=True,
        enable_logging=True,
        run_name=run_name,
        log_dir=LOG_DIR,
        record_trajectory=True,
        **env_kwargs
    )
    
    obs, _ = env.reset(seed=seed)
    # The MSSP agent handles its own rolling horizon internal compilation
    agent = RollingHorizonMSSPAgent(env.unwrapped, planning_horizon=10)
    
    episode_reward = 0
    for t in range(PLANNING_HORIZON):
        action = agent.get_action(t)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        if terminated or truncated:
            break
            
    metrics = _kpi(env.unwrapped, episode_reward)
    env.close()
    return metrics

def main():
    print(f"Starting Lightweight Benchmark to validate unified environment...")
    print(f"Testing Scenarios: {len(DEMAND_TYPES)} demand types x {len(GOODWILL_FLAGS)} goodwill flags x {NUM_SEEDS} seeds")
    print(f"Logging outputs to: {LOG_DIR}")
    
    os.makedirs(LOG_DIR, exist_ok=True)
    
    results = []
    
    for demand_type in DEMAND_TYPES:
        for use_goodwill in GOODWILL_FLAGS:
            for seed in tqdm(EVAL_SEEDS, desc=f"{demand_type} (GW={use_goodwill})"):
                
                env_kwargs = {
                    'scenario': SCENARIO,
                    'num_periods': PLANNING_HORIZON,
                    'demand_config': {
                        'type': demand_type,
                        'use_goodwill': use_goodwill
                    }
                }
                
                # 1. Run Heuristic
                try:
                    heur_metrics = run_heuristic(env_kwargs, seed)
                    results.append({
                        'agent': 'Heuristic',
                        'scenario': SCENARIO,
                        'demand': demand_type,
                        'goodwill': use_goodwill,
                        'seed': seed,
                        'profit': heur_metrics['profit']
                    })
                except Exception as e:
                    print(f"Error running Heuristic on seed {seed}: {e}")
                    
                # 2. Run MSSP
                try:
                    mssp_metrics = run_mssp(env_kwargs, seed)
                    results.append({
                        'agent': 'MSSP',
                        'scenario': SCENARIO,
                        'demand': demand_type,
                        'goodwill': use_goodwill,
                        'seed': seed,
                        'profit': mssp_metrics['profit']
                    })
                except Exception as e:
                    print(f"Error running MSSP on seed {seed}: {e}")

    df = pd.DataFrame(results)
    print("\n--- Lightweight Benchmark Results ---")
    print(df.groupby(['agent', 'demand', 'goodwill'])['profit'].mean().reset_index())
    
    # Save the consolidated scores
    out_csv = os.path.join(LOG_DIR, "lightweight_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved consolidated summary to {out_csv}")
    
    # Also verify that the logging wrapper dumped trajectories
    files_dumped = os.listdir(LOG_DIR)
    parquet_count = sum(1 for f in files_dumped if f.endswith('.parquet'))
    npz_count = sum(1 for f in files_dumped if f.endswith('.npz'))
    print(f"\nUniversalLoggingWrapper validation:")
    print(f"- .parquet files dumped (scalars): {parquet_count}")
    print(f"- .npz files dumped (trajectories): {npz_count}")

if __name__ == '__main__':
    main()
