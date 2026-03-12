import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import numpy as np
import time
from src.envs.core.environment import CoreEnv
from benchmark_iterative import _run_oracle, _acc, _mean, _std

# 1. Load the comprehensive results
df = pd.read_csv('data/results/benchmark_results_comprehensive_iterative.csv')

# 2. Identify target scenarios: Network='base' and Goodwill=True
# (We might want to do it for 'serial' network as well if Goodwill is evaluated there? The user specifically said "base network")
# "exclusively for the Goodwill = True scenarios on the base network"
mask = (df['Network'] == 'base') & (df['Goodwill'] == True)
target_rows = df[mask].copy()

print(f"Found {len(target_rows)} scenarios to recalculate for Oracle.")

max_episodes = 5
planning_horizon = 30

# 3. Re-evaluate
for idx, row in target_rows.iterrows():
    print(f"Recalculating: base | {row['Demand']} | gw=True | bl={row['Backlog']}...")
    env_kwargs = {
        'scenario': row['Network'],
        'backlog': bool(row['Backlog']),
        'demand_config': {'type': row['Demand'], 'use_goodwill': True, 'base_mu': 20},
        'num_periods': planning_horizon
    }
    
    metrics = {'profit': [], 'fill_rate': [], 'avg_inv': [], 'unfulfilled': []}
    
    for episode in range(max_episodes):
        seed = 42 + episode
        res = _run_oracle(env_kwargs, seed, planning_horizon)
        _acc(metrics, res)
        
    df.loc[idx, 'Oracle_Profit'] = _mean(metrics['profit'])
    df.loc[idx, 'Oracle_Profit_Std'] = _std(metrics['profit'])
    df.loc[idx, 'Oracle_FillRate'] = _mean(metrics['fill_rate'])
    df.loc[idx, 'Oracle_AvgInv'] = _mean(metrics['avg_inv'])
    df.loc[idx, 'Oracle_Unfulfilled'] = _mean(metrics['unfulfilled'])
    
    # Update VPI (Value of Perfect Information)
    mssp_profit = df.loc[idx, 'MSSP_Profit']
    df.loc[idx, 'VPI'] = _mean(metrics['profit']) - mssp_profit
    
    print(f"  -> Old Oracle Profit: {row['Oracle_Profit']:.2f} | New: {df.loc[idx, 'Oracle_Profit']:.2f}")

# 4. Save to a new file
output_file = 'benchmark_results_oracle_fixed.csv'
df.to_csv(output_file, index=False)
print(f"Successfully saved corrected matrix to {output_file}")
