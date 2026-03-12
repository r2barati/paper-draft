import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import argparse
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO

from src.envs.builder import make_supply_chain_env
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.ppo_gnn_agent import GNNFeaturesExtractor

# Full 32-Scenario Matrix
NETWORKS = ['base', 'serial']
DEMAND_TYPES = ['stationary', 'trend', 'seasonal', 'shock']
GOODWILL_FLAGS = [False, True]
BACKLOG_FLAGS = [True, False]

NUM_SEEDS = 10  # Reduced to 10 for speed during full 32-scenario sweep
EVAL_SEEDS = list(range(100, 100 + NUM_SEEDS))
PLANNING_HORIZON = 30
LOG_DIR = "./data/logs/unified_benchmark_rl"

def _kpi(env, episode_reward):
    total_u = np.sum(env.U)
    total_d = np.sum(env.D)
    fill_rate = max(0.0, 1.0 - (total_u / total_d)) if total_d > 0 else 1.0
    avg_inv = np.mean([np.sum(env.X[t]) for t in range(env.period)]) if env.period > 0 else 0
    return {
        'profit': episode_reward, # raw reward
        'avg_backlog': np.mean(np.sum(env.U, axis=1)),
        'fill_rate': fill_rate,
        'avg_inv': avg_inv,
        'final_sentiment': env.demand_engine.sentiment
    }

def run_rl_agent(agent_name, model_path, stats_path, env_kwargs, seed, custom_objects=None, use_residual=False):
    run_name = f"{agent_name}_{env_kwargs['scenario']}_{env_kwargs['demand_config']['type']}_gw{env_kwargs['demand_config']['use_goodwill']}_seed{seed}"
    
    # 1. Build the Unified Environment
    if use_residual:
        # We need a headless heuristic agent to feed the residual wrapper
        dummy_env = make_supply_chain_env(agent_type='or', **env_kwargs)
        heuristic = HeuristicAgent(dummy_env.unwrapped, is_blind=False)
        
        env = make_supply_chain_env(
            agent_type='residual_rl',
            heuristic_agent=heuristic,
            use_integer_actions=True, # The ResidualActionWrapper outputs floats, so integer enforcement is needed before it hits the core
            enable_logging=True,
            run_name=run_name,
            log_dir=LOG_DIR,
            record_trajectory=True, 
            **env_kwargs
        )
    else:
        env = make_supply_chain_env(
            agent_type='rl',
            use_integer_actions=True, # Convert continuous RL outputs [-1, 1] mapped to [0, max] back to integers
            enable_logging=True,
            run_name=run_name,
            log_dir=LOG_DIR,
            record_trajectory=True, 
            **env_kwargs
        )
    
    # 2. Load the PPO Model & VecNormalize Stats
    model = PPO.load(model_path, custom_objects=custom_objects)
    
    with open(stats_path, 'rb') as f:
        norm_env = pickle.load(f)
    norm_env.training = False
    norm_env.norm_reward = False

    # 3. Execution Loop
    obs, _ = env.reset(seed=seed)
    episode_reward = 0
    
    for t in range(PLANNING_HORIZON):
        # VecNormalize requires a 2D batch dimension
        # We slice from the END of the observation arrays to capture the domain features they were trained on
        expected_shape = norm_env.obs_rms.mean.shape[0]
        trimmed_obs = obs[-expected_shape:]
        
        obs_2d = trimmed_obs.reshape(1, -1)
        norm_obs = norm_env.normalize_obs(obs_2d).squeeze(0)
        
        # RL Predict
        action, _ = model.predict(norm_obs, deterministic=True)
        
        # SLICE FIX: Legacy models were trained on Base Network (size 11).
        # When evaluating on Serial Network (size 3), we must clip the output to avoid broadcast crashes.
        action = action[:len(env.unwrapped.network.reorder_links)]
        
        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        
        # We MUST use the raw profit from the environment, not any scaled reward
        episode_reward += info.get('raw_reward', reward) 
        
        if terminated or truncated:
            break
            
    metrics = _kpi(env.unwrapped, episode_reward)
    env.close() 
    return metrics

def generate_charts(df):
    plt.figure(figsize=(14, 10))
    sns.set_theme(style="whitegrid")
    
    # 1. Base Network Profit Comparison (Goodwill = True)
    df_base_gw = df[(df['scenario'] == 'base') & (df['goodwill'] == True) & (df['backlog'] == True)]
    plt.subplot(2, 1, 1)
    sns.barplot(data=df_base_gw, x='demand', y='profit', hue='agent', palette='Set1', ci=None)
    plt.title('Neural Network Performance: Base Network (Goodwill=True, Backlog=True)')
    plt.ylabel('Average Profit ($)')
    plt.xlabel('Demand Scenario')
    plt.legend(title='RL Agent Config')
    
    # 2. Serial Network Profit Comparison (Goodwill = False)
    df_serial_nogw = df[(df['scenario'] == 'serial') & (df['goodwill'] == False) & (df['backlog'] == True)]
    plt.subplot(2, 1, 2)
    sns.barplot(data=df_serial_nogw, x='demand', y='profit', hue='agent', palette='Set1', ci=None)
    plt.title('Neural Network Performance: Serial Network (Goodwill=False, Backlog=True)')
    plt.ylabel('Average Profit ($)')
    plt.xlabel('Demand Scenario')
    plt.legend(title='RL Agent Config')

    plt.tight_layout()
    chart_path = os.path.join(LOG_DIR, "rl_profit_comparison.png")
    plt.savefig(chart_path, dpi=300)
    print(f"\nGenerated Bar Chart Comparison: {chart_path}")
    
def main():
    print(f"Starting Highly Scalable Unified Benchmark for Neural Networks...")
    print(f"Sweeping 32 Scenarios: {len(NETWORKS)} Networks x {len(DEMAND_TYPES)} Demand x {len(GOODWILL_FLAGS)} GW x {len(BACKLOG_FLAGS)} Backlog")
    os.makedirs(LOG_DIR, exist_ok=True)
    
    results = []
    
    rl_agents = [
        {
            'name': 'RL_GNN',
            'model': 'data/models/ppo_gnn.zip',
            'stats': 'data/models/vec_normalize_gnn.pkl',
            'custom_objs': {"GNNFeaturesExtractor": GNNFeaturesExtractor},
            'is_residual': False
        },
        {
            'name': 'RL_Residual',
            'model': 'data/models/ppo_residual.zip',
            'stats': 'data/models/vec_normalize_residual.pkl',
            'custom_objs': None,
            'is_residual': True
        }
    ]
    
    # Start the 32-scenario sweep
    for network in NETWORKS:
        for backlog in BACKLOG_FLAGS:
            for demand_type in DEMAND_TYPES:
                for use_goodwill in GOODWILL_FLAGS:
                    
                    env_kwargs = {
                        'scenario': network,
                        'backlog': backlog,
                        'num_periods': PLANNING_HORIZON,
                        'demand_config': {
                            'type': demand_type,
                            'use_goodwill': use_goodwill
                        }
                    }
                    
                    scenario_str = f"[{network} | {demand_type} | GW={use_goodwill} | BL={backlog}]"
                    print(f"\nEvaluating Scenario: {scenario_str}")
                    
                    for agent_cfg in rl_agents:
                        # Before running a full inner loop, check if the model file physically exists
                        if not os.path.exists(agent_cfg['model']):
                            print(f"  Skipping {agent_cfg['name']}: Fast file check failed ({agent_cfg['model']} not found).")
                            continue
                            
                        for seed in tqdm(EVAL_SEEDS, desc=f"  -> {agent_cfg['name']}", leave=False):
                            try:
                                metrics = run_rl_agent(
                                    agent_name=agent_cfg['name'],
                                    model_path=agent_cfg['model'],
                                    stats_path=agent_cfg['stats'],
                                    env_kwargs=env_kwargs,
                                    seed=seed,
                                    custom_objects=agent_cfg['custom_objs'],
                                    use_residual=agent_cfg['is_residual']
                                )
                                results.append({
                                    'agent': agent_cfg['name'],
                                    'scenario': network,
                                    'backlog': backlog,
                                    'demand': demand_type,
                                    'goodwill': use_goodwill,
                                    'seed': seed,
                                    'profit': metrics['profit'],
                                    'avg_backlog': metrics['avg_backlog'],
                                    'fill_rate': metrics['fill_rate'],
                                    'avg_inv': metrics['avg_inv']
                                })
                            except Exception as e:
                                print(f"  Error mapping {agent_cfg['name']} Seed {seed}: {e}")
                                # Break the seed loop for this agent if there's a catastrophic error (e.g. shape mismatch)
                                break 

    df = pd.DataFrame(results)
    
    if df.empty:
         print("\nNo results were collected. Are the model files present?")
         return
         
    print("\n--- Summary Results (Average Profit) ---")
    summary = df.groupby(['agent', 'scenario', 'demand', 'goodwill'])['profit'].mean().unstack(level=[2, 3]).round(2)
    print(summary)
    
    out_csv = os.path.join(LOG_DIR, "rl_summary_comprehensive.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved raw results to {out_csv}")
    
    generate_charts(df)

if __name__ == '__main__':
    main()
