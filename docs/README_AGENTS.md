# Agent Controller Library

This document provides a deep dive into the various controller agents implemented in the `NetworkInvLog-BenchmarkTest` suite. Each agent acts as the "brain," dictating ordering policies for every node in the simulated supply chain environment at each time step.

## 1. Standalone Oracle Optimizer (`oracle.py` & `baselines.py`)
**Type:** Perfect Information / Mixed-Integer Linear Programming (MILP)
The Oracle acts as the ultimate upper bound for performance. It does not "guess" the future; it is provided with the full array of future demands for the entire episode upfront.

* **Mechanism (`StandaloneOracleOptimizer`)**: It translates the `gym` environment's topology, costs, and limits directly into PuLP objective functions and constraints. By solving the entire timeline simultaneously, it determines the globally optimal ordering path.
* **Mechanism (`EndogenousOracle`)**: For scenarios involving complex endogenous customer behaviors (`Goodwill=True`), standard MILP fails because demand becomes state-dependent. The `EndogenousOracle` solves this via iterative simulation-optimization heuristics to approximate the optimal bound.
* **Role**: Used as the comparative `y_true` standard. The benchmark calculates "Optimality Gaps" by comparing other agents' profits against the Oracle's profit.
* **Constraints Managed**: Flow conservation, non-negative inventory, ordering capacities, holding limits, and backlogged fulfillment logic.

## 2. Multi-Stage Stochastic Programming Agent (`mssp_agent.py`)
**Type:** Stochastic Optimization
The MSSP agent is the most mathematically robust "fair" agent. It knows the *distribution* of future demand but not the actual outcomes.

* **Mechanism**: At every step, it generates a probabilistic "scenario tree" (branching outward over its `planning_horizon`). It optimizes its current decision to perform the best *on average* across all potential future demand paths in the branch.
* **Computational Load**: Very high. The branching factor causes exponential growth in the equations it must solve via PuLP.
* **Parameters**: 
    * `planning_horizon`: How many steps into the future it looks.
    * `branching_depth`: The number of discrete probabilistic branches it maps at each future period.
    * `is_continuous`: Whether to relax integer constraints for speed.
    * `is_blind`: Forces the agent to act without complete state observability of downstream nodes.

## 3. Deterministic Linear Programming Agent (`dlp_agent.py`)
**Type:** Expected-Value Optimization
The DLP agent is a faster, simpler calculation alternative to the MSSP agent.

* **Mechanism**: Instead of mapping out multiple probability branches, it assumes future demand will exactly equal the expected average (the mean). It solves a linear programming model based on this single, deterministic future path over a rolling horizon.
* **Weakness**: It performs poorly in highly variable environments (like `shock` demands) because it fails to hedge against tail risks.

## 4. Heuristic Agent (`heuristic_agent.py`)
**Type:** Rule-Based / Classical Operations Research
This agent represents traditional supply chain practices used in industry before advanced AI/ML algorithms, derived directly from foundational Multi-Echelon literature.

* **Mechanism**: It implements a variant of a "Base Stock" or "Order-Up-To" policy. It statically calculates a target inventory level based on expected demand, safety factors, and lead times, natively ordering just enough to replenish to that target.
* **Role**: Serves as the realistic baseline for how a standard non-AI corporate entity would manage the supply chain today. Allows for "VPF_Heuristic" metric generation to measure the value of upgrading to AI systems.

## 5. Dummy / Random Agent (`benchmark.py`)
**Type:** Random Policy
* **Mechanism**: At every time step, it randomly samples actions uniformly from the environment's `action_space`.
* **Role**: Used purely to prove that the environment functioning logic is stable and to illustrate the absolute minimum floor of performance (which usually results in massive negative profits and complete system collapse).

## 6. Deep Reinforcement Learning (PPO) Agents (`rl_agent*.py`)
**Type:** Model-Free Deep Reinforcement Learning
A series of Proximal Policy Optimization (PPO) algorithms trained via `Stable-Baselines3` to navigate the complex multi-echelon environments using dense custom observation schemas.
* **V1 / V2 Architectures**: Standard PPO approaches leveraging multi-layer perceptrons, trained with significant Domain Randomization across capacities and demand distributions, alongside dense reward shaping algorithms (smoothing penalty drops).
* **V3 / V4 Architectures**: Advanced ablations stripping away domain randomization and shaping weights to investigate generalized "out-of-the-box" learning robustness. Relies heavily on accurate, normalized state encodings spanning the network topography.

## 7. Residual Reinforcement Learning Agent (`rl_agent_residual.py`)
**Type:** Hybrid Policy (Heuristic + Deep RL)
An advanced architecture designed to bridge the gap between classical OR safety and RL optimization.
* **Mechanism**: A deterministic `HeuristicAgent` calculates a baseline "safe" base-stock action vector for the current time step. A PPO model then observes the state and outputs a *residual* action (a $+/-$ modifier bound by the `action_space`), which is added to the heuristic policy. This provides the standard stability of traditional logistics augmented by AI micro-adjustments.

## 8. Graph Neural Network (GNN) Agent (`gnn_agent.py` & `gnn_extractor.py`)
**Type:** Topology-Aware Deep Reinforcement Learning
Designed to achieve true generalization beyond fixed input shapes by mathematically interpreting the network's echelon distances.
* **Mechanism**: Utilizes PyTorch Geometric to dynamically encode the underlying directed physical supply chain map alongside the respective node states (inventory, backlog). This allows the RL agent to adapt to entirely new routing topologies (`base` vs `serial`) or extended supply lines without requiring architectural retraining.
