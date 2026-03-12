import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import time
import numpy as np

from src.envs.core.environment import CoreEnv
from src.agents.baselines import EndogenousOracle
from benchmark_iterative import _kpi, _run_oracle

def test_endogenous():
    env_kwargs = {
        'scenario': 'base',
        'backlog': True,
        'demand_config': {'type': 'stationary', 'use_goodwill': True, 'base_mu': 20},
        'num_periods': 30
    }
    seed = 42

    print("Running standard Oracle (which is broken for Goodwill=True)...")
    res_standard = _run_oracle(env_kwargs, seed, 30)
    print(f"Standard Profit: {res_standard['profit']:.2f} (Fill Rate: {res_standard['fill_rate']:.3f})")

    print("\nRunning Endogenous Oracle Iterative Solver...")
    oracle = EndogenousOracle(env_kwargs, planning_horizon=30)
    
    t0 = time.time()
    best_actions = oracle.solve(seed)
    t1 = time.time()
    
    if best_actions is not None:
        # Evaluate
        eval_env = CoreEnv(**env_kwargs)
        eval_env.reset(seed=seed)
        ep_reward = 0
        for t in range(30):
            _, reward, _, _, _ = eval_env.step(best_actions[t])
            ep_reward += reward
            
        kpi = _kpi(eval_env, ep_reward)
        print(f"Converged Profit: {kpi['profit']:.2f} (Fill Rate: {kpi['fill_rate']:.3f})")
        print(f"Solved in {t1-t0:.2f}s")
    else:
        print("Convergence Failed.")

if __name__ == "__main__":
    test_endogenous()
