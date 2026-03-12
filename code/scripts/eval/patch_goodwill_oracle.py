import pandas as pd
import numpy as np
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.envs.core.environment import CoreEnv
from scripts.eval.benchmark_iterative import _run_oracle, _acc, _mean, _std

def main():
    # 1. Load the comprehensive results
    csv_file = 'data/results/benchmark_results_comprehensive_iterative.csv'
    df = pd.read_csv(csv_file)
    
    # 2. Identify target scenarios: Network='serial' and Goodwill=True
    mask = (df['Network'] == 'serial') & (df['Goodwill'] == True)
    target_rows = df[mask].copy()
    
    print(f"Found {len(target_rows)} scenarios to recalculate for Oracle.")
    
    max_episodes = 5
    planning_horizon = 30
    
    # 3. Re-evaluate
    for idx, row in target_rows.iterrows():
        print(f"Recalculating: serial | {row['Demand']} | gw=True | bl={row['Backlog']}...")

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
            
        old_prof = df.loc[idx, 'Oracle_Profit']
        df.loc[idx, 'Oracle_Profit'] = _mean(metrics['profit'])
        df.loc[idx, 'Oracle_Profit_Std'] = _std(metrics['profit'])
        df.loc[idx, 'Oracle_FillRate'] = _mean(metrics['fill_rate'])
        df.loc[idx, 'Oracle_AvgInv'] = _mean(metrics['avg_inv'])
        df.loc[idx, 'Oracle_Unfulfilled'] = _mean(metrics['unfulfilled'])
        
        # Update VPI (Value of Perfect Information)
        mssp_profit = df.loc[idx, 'MSSP_Profit']
        df.loc[idx, 'VPI'] = _mean(metrics['profit']) - mssp_profit
        
        print(f"  -> Old Oracle Profit: {old_prof:.2f} | New: {df.loc[idx, 'Oracle_Profit']:.2f}")
    
    # 4. Save to a new file
    output_file = csv_file
    df.to_csv(output_file, index=False)
    print(f"Successfully saved corrected matrix to {output_file}")

if __name__ == "__main__":
    main()
