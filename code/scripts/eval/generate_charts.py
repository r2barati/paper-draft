import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_data(csv_path):
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        return None

def process_data(df):
    """
    Process the dataframe to extract necessary columns for plotting.
    Specifically structure data to handle different agents and 'blind' configurations.
    """
    agents = ['Oracle', 'MSSP', 'DLP', 'Heuristic', 'Dummy', 'RLV1', 'RLV4', 'Residual', 'RLGNN']
    base_metrics = ['Profit', 'FillRate']
    
    # Restructure data for easier plotting (melt)
    rows = []
    
    for idx, row in df.iterrows():
        network = row['Network']
        demand = row['Demand']
        goodwill = row['Goodwill']
        backlog = row['Backlog']
        
        for agent in agents:
            # Check if agent has standard metrics
            profit_col = f"{agent}_Profit"
            fill_col = f"{agent}_FillRate"
            
            if profit_col in row and not pd.isna(row[profit_col]):
                rows.append({
                    'Network': network, 'Demand': demand, 'Goodwill': goodwill, 'Backlog': backlog,
                    'Agent': agent, 'Blind': False,
                    'Profit': row[profit_col], 'FillRate': row[fill_col]
                })
            
            # Check if agent has blind metrics
            blind_profit_col = f"{agent}_Blind_Profit"
            blind_fill_col = f"{agent}_Blind_FillRate"
            
            if blind_profit_col in row and not pd.isna(row[blind_profit_col]):
                rows.append({
                    'Network': network, 'Demand': demand, 'Goodwill': goodwill, 'Backlog': backlog,
                    'Agent': f"{agent} (Blind)", 'Blind': True,
                    'Profit': row[blind_profit_col], 'FillRate': row[blind_fill_col]
                })

    return pd.DataFrame(rows)


def plot_average_profit(processed_df, out_dir):
    """Plot average profit per agent across all scenarios."""
    plt.figure(figsize=(12, 8))
    sns.barplot(data=processed_df, x='Agent', y='Profit', errorbar=('ci', 95))
    plt.title('Average Achieved Profit by Agent (All Scenarios)', fontsize=16)
    plt.ylabel('Average Profit', fontsize=14)
    plt.xlabel('Agent', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '01_average_profit.png'))
    plt.close()
    
def plot_profit_by_network(processed_df, out_dir):
    plt.figure(figsize=(14, 8))
    sns.barplot(data=processed_df, x='Agent', y='Profit', hue='Network', errorbar=None)
    plt.title('Average Profit by Agent and Network Topology', fontsize=16)
    plt.ylabel('Average Profit', fontsize=14)
    plt.xlabel('Agent', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Network')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '02_profit_by_network.png'))
    plt.close()

def plot_fill_rate(processed_df, out_dir):
    """Plot average fill rate per agent."""
    plt.figure(figsize=(12, 8))
    sns.barplot(data=processed_df, x='Agent', y='FillRate', errorbar=('ci', 95))
    plt.title('Average Service Level (Fill Rate) by Agent', fontsize=16)
    plt.ylabel('Average Fill Rate', fontsize=14)
    plt.xlabel('Agent', fontsize=14)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '03_average_fill_rate.png'))
    plt.close()

def plot_optimality_gap(df, out_dir):
    """Plot Optimality Gap relative to Oracle."""
    agents_to_compare = ['MSSP', 'DLP', 'Heuristic', 'Dummy', 'RLV1', 'RLV4', 'Residual', 'RLGNN']
    gap_rows = []
    
    for idx, row in df.iterrows():
        oracle_profit = row.get('Oracle_Profit', pd.NA)
        if pd.isna(oracle_profit) or oracle_profit <= 0:
            continue # Skip negative or zero oracle profits for gap percentage or handle differently?
            # Let's use absolute difference for robustness if negative profits exist
            
        for agent in agents_to_compare:
            profit_col = f"{agent}_Profit"
            if profit_col in row and not pd.isna(row[profit_col]):
                # Absolute gap: Oracle Profit - Agent Profit
                gap = oracle_profit - row[profit_col]
                # Relative gap: (Oracle - Agent) / |Oracle| 
                rel_gap = gap / abs(oracle_profit) if oracle_profit != 0 else 0
                
                gap_rows.append({
                    'Agent': agent,
                    'Network': row['Network'],
                    'Demand': row['Demand'],
                    'Absolute_Gap': gap,
                    'Relative_Gap_Pct': rel_gap * 100
                })
                
    gap_df = pd.DataFrame(gap_rows)
    
    if not gap_df.empty:
        plt.figure(figsize=(12, 8))
        sns.barplot(data=gap_df, x='Agent', y='Relative_Gap_Pct', errorbar=('ci', 95))
        plt.title('Relative Optimality Gap to Oracle (%)', fontsize=16)
        plt.ylabel('Optimality Gap (%)', fontsize=14)
        plt.xlabel('Agent', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '04_optimality_gap_relative.png'))
        plt.close()

def plot_information_asymmetry(df, out_dir):
    """Plot Value of Perfect Information (VPI) and Value of Stochastic Solution (VSS)."""
    # Assuming VPI and VSS columns exist as per schema
    metrics = ['VPI', 'VSS']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if available_metrics:
        melted = df.melt(id_vars=['Network', 'Demand'], value_vars=available_metrics, var_name='Metric', value_name='Value')
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=melted, x='Metric', y='Value')
        plt.title('Information Value Metrics (VPI & VSS) Distribution', fontsize=16)
        plt.ylabel('Value', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '05_info_asymmetry.png'))
        plt.close()

def plot_blind_vs_aware(processed_df, out_dir):
    """Plot performance difference between Blind vs Aware configurations for relevant agents."""
    agents_with_blind = [a.replace(' (Blind)', '') for a in processed_df[processed_df['Blind'] == True]['Agent'].unique()]
    
    if not agents_with_blind:
        return
        
    filtered_df = processed_df[processed_df['Agent'].apply(lambda x: x.replace(' (Blind)', '') in agents_with_blind)]
    
    # We need to map 'Agent' back to base agent + blind status for hue
    filtered_df['Base_Agent'] = filtered_df['Agent'].apply(lambda x: x.replace(' (Blind)', ''))
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=filtered_df, x='Base_Agent', y='Profit', hue='Blind', errorbar=None)
    plt.title('Impact of Information Asymmetry (Aware vs Blind)', fontsize=16)
    plt.ylabel('Average Profit', fontsize=14)
    plt.xlabel('Agent', fontsize=14)
    plt.legend(title='Is Blind?')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '06_blind_vs_aware.png'))
    plt.close()
    
def plot_rl_training_curves(log_dir, out_dir):
    # Optional logic if we want to plot RL training curves, we skip for now since task implies CSV benchmark data
    pass

def main():
    sns.set_theme(style="whitegrid")
    
    input_csv = "data/results/benchmark_results_comprehensive_iterative.csv"
    out_dir = "benchmark_charts"
    
    ensure_dir(out_dir)
    
    df = load_data(input_csv)
    if df is not None:
        processed_df = process_data(df)
        
        print("Generating Average Profit Chart...")
        plot_average_profit(processed_df, out_dir)
        
        print("Generating Profit by Network Chart...")
        plot_profit_by_network(processed_df, out_dir)
        
        print("Generating Fill Rate Chart...")
        plot_fill_rate(processed_df, out_dir)
        
        print("Generating Optimality Gap Chart...")
        plot_optimality_gap(df, out_dir)
        
        print("Generating Information Asymmetry Chart...")
        plot_information_asymmetry(df, out_dir)
        
        print("Generating Blind vs Aware Chart...")
        plot_blind_vs_aware(processed_df, out_dir)
        
        print(f"All charts successfully generated in '{out_dir}/'")

if __name__ == "__main__":
    main()
