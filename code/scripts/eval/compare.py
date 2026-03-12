import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import time
from src.envs.core.environment import CoreEnv
from src.agents.oracle import StandaloneOracleOptimizer

def run_comparison():
    N_EPISODES = 100
    profits = []
    
    start_time = time.time()
    
    for i in range(N_EPISODES):
        seed = i
        
        env_kwargs = {
            'scenario': 'base',
            'backlog': True,
            'demand_config': {
                'type': 'stationary',
                'use_goodwill': False,
                'base_mu': 20
            },
            'num_periods': 30
        }
        
        # N-2 fix: use a zero-action probe episode to extract the exact realized
        # demand trace (same pattern as benchmark.py evaluate_oracle).
        # This guarantees the Oracle plans over the same demand path that the
        # simulation will realize, eliminating the separate-RNG divergence.
        probe_env = CoreEnv(**env_kwargs)
        probe_env.reset(seed=seed)
        for _ in range(30):
            probe_env.step(np.zeros(probe_env.action_space.shape))
        
        demand_trace = {}
        for idx, edge in enumerate(probe_env.network.retail_links):
            demand_trace[edge] = probe_env.D[:, idx].copy()
        
        # Pin the extracted demand as user_D so both env and Oracle see the same path
        fixed_kwargs = env_kwargs.copy()
        fixed_kwargs['user_D'] = demand_trace
        
        env_sim = CoreEnv(**fixed_kwargs)
        env_sim.reset(seed=seed)
        
        # 3. Initialize Oracle (LP relaxation, matching benchmark.py)
        oracle = StandaloneOracleOptimizer(env_sim, demand_trace, planning_horizon=30, is_continuous=True)
        
        # 4. Solve
        optimal_actions = oracle.solve_full_horizon()
        
        if optimal_actions is None:
            continue
            
        episode_reward = 0
        for t in range(30):
            action = optimal_actions[t]
            _, reward, _, _, _ = env_sim.step(action)
            episode_reward += reward
            
        profits.append(episode_reward)
        
        if (i+1) % 10 == 0:
            print(f"Completed {i+1}/{N_EPISODES}...")
            
    elapsed = time.time() - start_time
    
    mean_profit = np.mean(profits)
    std_profit = np.std(profits)
    
    print("\n============================================================")
    print("NEW COMPREHENSIVE ORACLE RESULTS (N=100)")
    print("============================================================")
    print(f"Mean Profit: {mean_profit:.2f}")
    print(f"Std Deviation: {std_profit:.2f}")
    print(f"Total Time: {elapsed:.2f} seconds")
    print("============================================================")

if __name__ == '__main__':
    run_comparison()
