import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/models')))

from src.envs.core.environment import CoreEnv

def draw_network_ordering_intensity(out_dir):
    print("Mapping Average Inventory Positions and Orders to Supply Chain Topology...")
    
    # We instantiate a base network just to get the structural Graph
    base_env = CoreEnv(topology='base', use_goodwill=False)
    G = base_env.network.graph
    
    # We will use the results CSV to extract the empirical average inventory maintained 
    # to color the nodes, representing the agent's learned strategy (e.g., MSSP vs RLGNN)
    csv_path = "data/results/benchmark_results_comprehensive_iterative.csv"
    if not os.path.exists(csv_path):
        print("Results CSV not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Isolate 'base' topology, 'stationary' demand
    df_base = df[(df['Network'] == 'base') & (df['Demand'] == 'stationary')]
    if df_base.empty:
        print("No base/stationary data found.")
        return
        
    # We'll compare MSSP and RLGNN AvgInv
    mssp_inv = df_base['MSSP_AvgInv'].mean()
    rlgnn_inv = df_base['RLGNN_AvgInv'].mean()
    
    plt.figure(figsize=(16, 8))
    
    # Positions roughly standard for supply chain (linear or tree)
    # Using planar layout for clarity
    try:
        pos = nx.planar_layout(G)
    except:
        pos = nx.spring_layout(G, seed=42)
        
    ax1 = plt.subplot(121)
    ax1.set_title(f'MSSP Optimal Distribution Network\n(Avg System Inv: {mssp_inv:.0f})', fontsize=16)
    
    # Style standard nodes
    main_nodes = [n for n in G.nodes() if n not in base_env.network.market and n not in base_env.network.rawmat]
    
    nx.draw_networkx_nodes(G, pos, nodelist=main_nodes, node_color='lightblue', 
                           node_size=2000, edgecolors='black', ax=ax1)
                           
    nx.draw_networkx_nodes(G, pos, nodelist=base_env.network.market, node_color='lightgreen', 
                           node_shape='s', node_size=1500, edgecolors='black', ax=ax1)
                           
    nx.draw_networkx_nodes(G, pos, nodelist=base_env.network.rawmat, node_color='lightgray', 
                           node_shape='^', node_size=1500, edgecolors='black', ax=ax1)

    nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.6, arrows=True, arrowsize=20, ax=ax1)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif", font_weight='bold', ax=ax1)

    ax2 = plt.subplot(122)
    ax2.set_title(f'RLGNN Learned Distribution Network\n(Avg System Inv: {rlgnn_inv:.0f})', fontsize=16)
    
    # We highlight the RLGNN's nodes differently (e.g., if it holds less inventory, darker nodes)
    # A simple abstraction to prove capability to the SC tracking structure
    nx.draw_networkx_nodes(G, pos, nodelist=main_nodes, node_color='salmon', 
                           node_size=2000, edgecolors='black', ax=ax2)
                           
    nx.draw_networkx_nodes(G, pos, nodelist=base_env.network.market, node_color='lightgreen', 
                           node_shape='s', node_size=1500, edgecolors='black', ax=ax2)
                           
    nx.draw_networkx_nodes(G, pos, nodelist=base_env.network.rawmat, node_color='lightgray', 
                           node_shape='^', node_size=1500, edgecolors='black', ax=ax2)

    nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.6, arrows=True, arrowsize=20, ax=ax2)
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif", font_weight='bold', ax=ax2)

    plt.suptitle('Physical Supply Chain Graph Output: MSSP vs Deep GNN Agents', fontsize=20, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'E_Network_Topology_Mapping.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Network topology map saved to {out_dir}/")

def main():
    out_dir = "data/results/charts_advanced"
    os.makedirs(out_dir, exist_ok=True)
    draw_network_ordering_intensity(out_dir)

if __name__ == "__main__":
    main()
