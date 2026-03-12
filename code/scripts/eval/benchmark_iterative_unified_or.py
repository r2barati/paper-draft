import os
import argparse
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.envs.builder import make_supply_chain_env
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.mssp_agent import RollingHorizonMSSPAgent
from src.agents.dlp_agent import RollingHorizonDLPAgent
from src.agents.baselines import OptimisticEndogenousOracle

SCENARIO = 'base'
DEMAND_TYPES = ['stationary', 'trend', 'seasonal', 'shock']
GOODWILL_FLAGS = [False, True]
NUM_SEEDS = 30
EVAL_SEEDS = list(range(100, 100 + NUM_SEEDS))
PLANNING_HORIZON = 30
LOG_DIR = "./data/logs/unified_benchmark_or"

def _kpi(env, episode_reward):
    return {
        'profit': np.sum(env.P),
        'avg_backlog': np.mean(np.sum(env.U, axis=1)),
        'final_sentiment': env.demand_engine.sentiment
    }

def run_agent(agent_name, agent_class, env_kwargs, seed, lookahead=None):
    run_name = f"{agent_name}_{env_kwargs['scenario']}_{env_kwargs['demand_config']['type']}_gw{env_kwargs['demand_config']['use_goodwill']}_seed{seed}"
    
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
    
    if agent_name == "Heuristic":
        agent = agent_class(env.unwrapped, is_blind=False)
    elif agent_name == "Oracle":
        optimizer = OptimisticEndogenousOracle(env.unwrapped, env.unwrapped.num_periods)
        action_matrix = optimizer.solve()
    elif lookahead is not None:
        agent = agent_class(env.unwrapped, planning_horizon=lookahead)
        
    episode_reward = 0
    for t in range(PLANNING_HORIZON):
        if agent_name == "Oracle":
            action = action_matrix[t]
        else:
            action = agent.get_action(t) if hasattr(agent, 'get_action') and 't' in agent.get_action.__code__.co_varnames else agent.get_action(obs, t)
            
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        if terminated or truncated:
            break
            
    metrics = _kpi(env.unwrapped, episode_reward)
    env.close() 
    return metrics

def generate_charts(df):
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    # 1. Profit Distribution by Agent and Demand Type (No Goodwill)
    df_no_gw = df[df['goodwill'] == False]
    plt.subplot(2, 1, 1)
    sns.boxplot(data=df_no_gw, x='demand', y='profit', hue='agent', palette='Set2')
    plt.title('Profit Distribution by Baseline Model (Goodwill = False)')
    plt.ylabel('Total Profit ($)')
    plt.xlabel('Demand Scenario')
    
    # 2. Profit Distribution by Agent and Demand Type (Goodwill = True)
    df_gw = df[df['goodwill'] == True]
    plt.subplot(2, 1, 2)
    sns.boxplot(data=df_gw, x='demand', y='profit', hue='agent', palette='Set2')
    plt.title('Profit Distribution by Baseline Model (Goodwill = True)')
    plt.ylabel('Total Profit ($)')
    plt.xlabel('Demand Scenario')
    
    plt.tight_layout()
    chart_path = os.path.join(LOG_DIR, "baseline_profit_comparison.png")
    plt.savefig(chart_path, dpi=300)
    print(f"\nGenerated Boxplot Comparison: {chart_path}")
    
def main():
    print(f"Starting Unified Benchmark for OR Baselines...")
    os.makedirs(LOG_DIR, exist_ok=True)
    
    results = []
    
    # Configuration
    agents = [
        ("Heuristic", HeuristicAgent, None),
        ("DLP", RollingHorizonDLPAgent, 10),
        ("MSSP", RollingHorizonMSSPAgent, 10),
        ("Oracle", OptimisticEndogenousOracle, None)
    ]
    
    for agent_name, agent_class, lookahead in agents:
        print(f"\nTesting {agent_name} Agent...")
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
                    try:
                        metrics = run_agent(agent_name, agent_class, env_kwargs, seed, lookahead)
                        results.append({
                            'agent': agent_name,
                            'scenario': SCENARIO,
                            'demand': demand_type,
                            'goodwill': use_goodwill,
                            'seed': seed,
                            'profit': metrics['profit'],
                            'avg_backlog': metrics['avg_backlog'],
                            'final_sentiment': metrics['final_sentiment']
                        })
                    except Exception as e:
                        print(f"Error compiling {agent_name} | {demand_type} | GW={use_goodwill} | Seed={seed}: {e}")

    df = pd.DataFrame(results)
    
    print("\n--- Summary Results ---")
    summary = df.groupby(['agent', 'demand', 'goodwill'])['profit'].mean().unstack(level=[1, 2]).round(2)
    print(summary)
    
    out_csv = os.path.join(LOG_DIR, "baseline_summary_comprehensive.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved raw results to {out_csv}")
    
    generate_charts(df)

if __name__ == '__main__':
    main()
