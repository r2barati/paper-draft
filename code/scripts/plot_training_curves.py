import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

sns.set_theme(style="whitegrid")
sns.set_context("talk")

def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)

def parse_npz_evaluations(log_dir):
    """Parses standard stable-baselines3 evaluations.npz file."""
    eval_file = os.path.join(log_dir, 'evaluations.npz')
    if not os.path.exists(eval_file):
        print(f"[Warning] No evaluations.npz found in {log_dir}")
        return None, None, None
        
    data = np.load(eval_file)
    timesteps = data['timesteps']
    results = data['results'] # shape (n_evals, n_eval_envs)
    
    # Calculate mean and std over eval envs
    mean_rewards = np.mean(results, axis=1)
    std_rewards = np.std(results, axis=1)
    
    return timesteps, mean_rewards, std_rewards

def plot_learning_curves(log_dirs_dict, out_dir):
    """
    Plots learning curves for multiple agents on the same figure.
    log_dirs_dict: dict mapping 'Agent Name' -> 'path/to/log_dir'
    """
    plt.figure(figsize=(14, 8))
    
    plotted_any = False
    for agent_name, log_dir in log_dirs_dict.items():
        timesteps, mean_rewards, std_rewards = parse_npz_evaluations(log_dir)
        if timesteps is not None:
            plt.plot(timesteps, mean_rewards, label=agent_name, linewidth=2)
            plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2)
            plotted_any = True
            
    if not plotted_any:
        print("No valid evaluation data found to plot learning curves.")
        return
        
    plt.title('Deep RL & GNN Learning Curves (Evaluation Rewards)', fontsize=18, pad=20)
    plt.xlabel('Training Timesteps', fontsize=14)
    plt.ylabel('Mean Evaluation Reward (Smoothed Profit)', fontsize=14)
    plt.legend(title='Architecture', loc='lower right')
    
    # Optionally format x-axis to be more readable (e.g., 1M)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{int(x/1000)}k' if x >= 1000 else int(x)))
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'LC_multi_agent_convergence.png'), dpi=300)
    plt.close()

def main():
    out_dir = "data/logs/charts"
    ensure_dir(out_dir)
    
    # Define the mapping of agent names to their respective log directories
    log_dirs = {
        'PPO (V2)': 'data/logs/ppo_v2_logs',
        'PPO (V3)': 'data/logs/ppo_v3_logs',
        'PPO (V4)': 'data/logs/ppo_v4_logs',
        'PPO-Residual': 'data/logs/ppo_residual_logs',
        'PPO-GNN': 'data/logs/ppo_gnn_logs'
    }
    
    print("Parsing evaluation logs and plotting learning curves...")
    plot_learning_curves(log_dirs, out_dir)
    print(f"Learning curves saved to {out_dir}/")

if __name__ == "__main__":
    main()
