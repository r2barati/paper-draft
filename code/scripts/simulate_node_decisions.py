import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from stable_baselines3 import PPO

# Import project environment components
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/models')))

from src.envs.core.environment import CoreEnv
from src.envs.wrappers.feature_wrappers import DomainFeatureWrapper
from src.envs.wrappers.action_wrappers import ResidualActionWrapper
from src.agents.heuristic_agent import HeuristicAgent

sns.set_theme(style="whitegrid")

def simulate_and_plot(model_path, agent_type, out_dir):
    print(f"Loading {agent_type} from {model_path}...")
    
    # 1. Instantiate the base environment (Standard Evaluation Config)
    base_env = CoreEnv(
        topology='base', 
        demand_type='stationary', 
        use_goodwill=False, 
        is_blind=False,
        max_episodes=1, 
        render_mode=None
    )
    
    # 2. Wrap based on agent type
    if agent_type == 'Residual':
        heuristic_base = HeuristicAgent(base_env, is_blind=False)
        env = ResidualRLWrapper(base_env, heuristic_base, is_blind=False)
    else:
        env = DomainFeatureWrapper(base_env, is_blind=False)
        
    # main_nodes from wrapper
    num_nodes = len(env.main_nodes)
    
    # 3. Load Model
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # 4. Storage for simulation data
    observation, info = env.reset(seed=42)
    
    timesteps = []
    actions_taken = {i: [] for i in range(num_nodes)}
    inventory_levels = {i: [] for i in range(num_nodes)}
    
    done = False
    truncated = False
    step = 0
    
    print("Simulating 1 full episode to trace node decisions...")
    while not (done or truncated):
        # Predict action
        action, _states = model.predict(observation, deterministic=True)
        
        # Log before step
        for i, node in enumerate(env.main_nodes):
            node_idx = base_env.network.node_map[node]
            actions_taken[i].append(action[i])
            inventory_levels[i].append(base_env.X[step, node_idx])
            
        timesteps.append(step)
        
        # Step environment
        observation, reward, done, truncated, info = env.step(action)
        step += 1

    # 5. Plot the actions taken by each node over time
    plt.figure(figsize=(14, 8))
    for i, node in enumerate(env.main_nodes):
        plt.plot(timesteps, actions_taken[i], label=f'Node {node} (Order Policy)', linewidth=2)
        
    plt.title(f'{agent_type} Decision Trajectory over 1 Episode (Stationary Demand)', fontsize=16, pad=20)
    plt.xlabel('Simulation Day (Timestep)', fontsize=14)
    plt.ylabel('Normalized Order Action [-1 to 1]', fontsize=14)
    plt.legend(title='Echelon Node')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'NODE_ACTIONS_{agent_type}.png'), dpi=300)
    plt.close()
    print(f"Saved node trajectory plot to {out_dir}/NODE_ACTIONS_{agent_type}.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='RLGNN', choices=['RLV2', 'RLV3', 'RLV4', 'Residual', 'RLGNN'])
    args = parser.parse_args()
    
    out_dir = "data/logs/charts"
    os.makedirs(out_dir, exist_ok=True)
    
    # Map agent name to likely best_model path
    model_paths = {
        'RLV2': 'data/logs/ppo_v2_logs/best_model.zip',
        'RLV3': 'data/logs/ppo_v3_logs/best_model.zip',
        'RLV4': 'data/logs/ppo_v4_logs/best_model.zip',
        'Residual': 'data/logs/ppo_residual_logs/best_model.zip',
        'RLGNN': 'data/logs/ppo_gnn_logs/best_model.zip'
    }
    
    target_path = model_paths[args.agent]
    if os.path.exists(target_path):
        simulate_and_plot(target_path, args.agent, out_dir)
    else:
        print(f"Error: Model file {target_path} not found.")

if __name__ == "__main__":
    main()
