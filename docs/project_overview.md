# Network Inventory Optimization Benchmark

![Python Status](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Supported-orange.svg)
![Optimization](https://img.shields.io/badge/Optimization-PuLP%20%7C%20SciPy-brightgreen.svg)

This repository provides a robust, highly configurable Digital Twin environment for multi-echelon supply chain networks, distributed alongside a comprehensive benchmark suite designed to evaluate classical Operations Research (OR) models, Deep Reinforcement Learning (DRL) algorithms, and custom heuristic algorithms under dynamic, non-stationary demand scenarios. 

This project evolved from a monolithic Jupyter notebook pipeline into an organized, modular, high-fidelity Simulation and Benchmarking Rig.

---

## 🎯 Project Objectives

1. **Abstracted Environment Simulation**: Isolate network topology, logistics mechanics, and demand generation into an independent, universally compatible Gymnasium (`gym`) environment. This allows seamless experimentation with varying supply chain shapes, lost sales vs. backlog logic, and customer behavioral dynamics (e.g., goodwill).
2. **Standardized Benchmarking Matrix**: Establish an automated testing rig to iterate through a large matrix of configurable environmental profiles, quantifying an agent's or algorithm's real-time performance simultaneously across dozens of scenarios.
3. **Fair Algorithmic Evaluation**: Pit different families of optimization models (Deterministic, Stochastic, Heuristic, Perfect Information) and cutting-edge Deep Reinforcement Learning (DRL) algorithms against one another on an equal footing, tracking both informed operations and "blind" operation scenarios.

---

## 📚 Deep Dive Documentation

For detailed coverage of specific components in this repository, please refer to the following comprehensive guides:
* [**Environment & Digital Twin Guide** (`README_ENVIRONMENT.md`)](README_ENVIRONMENT.md): Deep dive into topology, customer behaviors, and Gym mechanisms.
* [**Agent Controller Library** (`README_AGENTS.md`)](README_AGENTS.md): Detailed explanations of the Oracle, MSSP, DLP, and Heuristic decision-making models.
* [**Benchmark Data & Output Metrics** (`README_DATA.md`)](README_DATA.md): Comprehensive schema mapping of all generated CSV files (`benchmark_results_comprehensive.csv`, etc.) and "Value of Information" metrics.

---

## 🛠 Repository Architecture

The codebase abstracts supply chain logic, agents, and evaluations into highly modular Python scripts:

### 1. The Digital Twin (`environment.py`)
Contains the underlying physics, rules, topologies, and logic of the network:
- **`SupplyChainNetwork`**: Generates topology mappings, handling diverse configurations ranging from linear `serial` pipelines to complex networked `base` topologies.
- **`DemandEngine`**: Generates multi-period non-stationary demand trajectories (`stationary`, `trend`, `seasonal`, `shock`) and supports endogenous `goodwill` features, turning customer satisfaction into an active demand multiplier.
- **`NetInvMgmtMasterEnv_New`**: The orchestrating `gym.Env` coordinator. It standardizes logistics, delay pipelines, operating costs, and order executions. It natively handles varying fulfillment models (e.g., `backlog=True/False`).

### 2. The Agent Library
The repository includes several families of decision-making controllers. Each agent attempts to optimize ordering policies to maximize profit across the network.
- **`oracle.py` (`StandaloneOracleOptimizer` / `EndogenousOracle`)**: The globally-optimal baseline. Utilizes perfect future information and Mixed-Integer Linear Programming (MILP) to compute the best possible outcome. `EndogenousOracle` handles complex state-dependent customer goodwill logic.
- **`mssp_agent.py` (`RollingHorizonMSSPAgent`)**: A Multi-Stage Stochastic Programming (MSSP) agent drawing scenario trees to hedge against uncertainty over a rolling planning horizon.
- **`dlp_agent.py` (`RollingHorizonDLPAgent`)**: A Deterministic Linear Programming (DLP) agent using expected-value deterministic rolling horizon approximations.
- **`heuristic_agent.py` (`HeuristicAgent`)**: Traditional Operations Research agent running classical generalized rules (e.g., Base Stock policies) dynamically computed against target service levels.
- **`rl_agent_v2.py` / `gnn_agent.py` / `rl_agent_residual.py`**: State-of-the-art Deep Reinforcement Learning agents mapped via Stable-Baselines3, featuring custom multi-echelon observation wrappers, domain-randomized training pipelines, Graph Neural Networks (GNNs), and hybrid Residual RL combinations for robust non-stationary policy adoption.

*Note: OR Agents feature tunable parameters such as `is_blind` (reducing state visibility) and variable `planning_horizon` depths to ablate their information advantages.*

### 3. Execution & Testing Suite
- **`benchmark_iterative.py`**: The modernized, parallelized core evaluation script safely evaluating all models side-by-side using robust multiprocess file locking and incremental caching mechanisms (`*_cache.csv`). Maps a massive evaluation matrix spanning `base`/`serial` topologies, 4 demand profiles, `goodwill`, and `backlog` logic without crashing during long-running tasks.
- **`benchmark.py` (`BenchmarkSuite`)**: The legacy automated mapping evaluation suite. 
- **`compare.py`**: A specialized comparison script for smaller-scale baseline validations.

### 4. Data Artifacts & Results Tracker
Running the benchmarks yields comprehensive statistical snapshots across scenarios:
- **`benchmark_results_comprehensive_iterative.csv`**: Massive unified output mapping detailing scenario features against the concurrent performance of Oracle, MSSP, DLP, Heuristic, Dummy, and all DRL Agents (V1, V2, V4, GNN, Residual).
- **`benchmark_results_oracle_fixed.csv`**: The explicitly repaired evaluation set isolating perfect-information solutions in complex Endogenous Goodwill environments.
- **`ablation_study_summary.csv`**: Captures findings across varying horizons or parameter drops inside specific agent testing.

### 5. Research & Documentation Notebook
- **`V2 RL (PPO) - RL, Solutions...ipynb`**: The original comprehensive research notebook. It contains initial deep reinforcement learning integrations mapping Proximal Policy Optimization (PPO), deep OR implementations, deep-dive explorations into multi-echelon environments, relevant literature reviews, and legacy baseline comparisons.

---

## 🚀 Quickstart & Pipeline Usage

**1. Install Dependencies**
```bash
pip install gymnasium numpy pandas networkx scipy pulp
```

**2. Running the Comprehensive Benchmarking Matrix**
To trigger the automated multi-model scenario run, utilizing the cached multiprocess runner:
```bash
python benchmark_iterative.py
```
*Depending on the complexity of the MSSP and Oracle agents, computing the entire matrix across multiple random seeds will take substantial processing time. Results are safely iteratively cached as they complete.*

**3. Running Legacy Verifications**
To test deterministic logic or debug an isolated multi-period stationary test:
```bash
python compare.py
```

---

## 📊 Evaluation & Metrics Generated

Under the standard run, the benchmarking output data includes metrics such as:
* **Algorithm Achieved Profit**: Directly tied to the complex penalty, holding, and ordering cost dynamics across the chain.
* **Service Level Tracking**: Monitoring backlogged units or missed opportunities throughout all multi-echelon nodes.
* **Relative Efficiency (Optimality Gap)**: Evaluates actual profit captured vs. the `Oracle` upper-bound benchmark.
* **Information Asymmetry Impacts**: Captured delta between normal agents vs. `is_blind=True` scenarios exploring partial observability limitations.
