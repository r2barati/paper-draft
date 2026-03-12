# Research Log: Towards a Standard Benchmark for Inventory Management Methods

> Comprehensive record of all design decisions, experiments, and findings throughout this project.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Environment Design Decisions](#2-environment-design-decisions)
3. [Agent Development Timeline](#3-agent-development-timeline)
4. [Benchmark Experiment History](#4-benchmark-experiment-history)
5. [Paper Writing Process](#5-paper-writing-process)
6. [Key Technical Discussions](#6-key-technical-discussions)
7. [Unresolved Questions & Future Directions](#7-unresolved-questions--future-directions)

---

## 1. Project Overview

**Objective:** Build an open benchmarking framework that enables fair, reproducible comparison of inventory management methods across four paradigms: heuristic policies, mathematical programming, reinforcement learning, and deep learning.

**Core Contribution:** Rather than creating a new simulation, we extended the OR-Gym framework (Hubbs et al., 2020) with:
- Composable non-stationary demand patterns (stationary, trend, seasonal, shock)
- Endogenous goodwill dynamics (demand–service quality feedback loops)
- Multiple network topologies (base 4-echelon, serial)
- Classical OR methods implemented as first-class Gymnasium agents
- 10 solution methods evaluated across 32+ controlled scenarios

**Research Question:** Under what conditions do RL/DL methods outperform classical OR baselines, and vice versa?

---

## 2. Environment Design Decisions

### 2.1 Why Extend OR-Gym Instead of Building From Scratch?

**Discussion:** Early in the project, we debated whether to build a custom supply chain simulator or extend an existing one. The decision to extend OR-Gym was driven by:
- **Reproducibility:** Building on a published, peer-reviewed environment means our results are comparable to prior work.
- **Credibility:** OR-Gym is already cited in the RL-for-SCM literature.
- **Effort:** Extending is faster than building from scratch, allowing more time for agent development and experiments.

**Changes made to OR-Gym:**
- Migrated from legacy `gym` to `gymnasium` API (required for SB3 v2+)
- Refactored monolithic environment into modular `core/` (environment, demand_engine, network_topology) and `wrappers/` (feature, action, logging)
- Added configurable demand patterns via `demand_engine.py`
- Added goodwill dynamics as an optional toggle

### 2.2 Demand Pattern Design

**Discussion:** We debated how many demand patterns to include. The final four were chosen to test distinct algorithmic capabilities:

| Pattern | Tests | Why It Matters |
|---------|-------|----------------|
| Stationary | Baseline steady-state performance | Most classical theory assumes this |
| Trend | Adaptation to distributional shift | Tests if agents can track changing mean |
| Seasonal | Anticipatory ordering | Tests if agents can plan ahead for peaks |
| Shock | Resilience to sudden disruption | Tests robustness to regime change |

**Key decision:** Demand patterns are composable—the `demand_engine.py` accepts a callable that transforms the base rate. This means researchers can define new patterns without modifying the environment.

### 2.3 Goodwill Dynamics

**Discussion:** A major design discussion was whether to include endogenous demand dynamics. The goodwill mechanism was added because:
- Real-world demand is not exogenous—poor service causes customer churn
- It creates a clear differentiator between OR methods (which assume exogenous demand) and RL methods (which can learn the feedback loop)
- It tests whether RL provides genuine value over classical methods

**Implementation:** The goodwill multiplier $G_t \in [0.5, 1.5]$ evolves as a function of the fill rate. High fill rate → goodwill increases → higher future demand. Low fill rate → goodwill decreases → lower future demand. This creates a virtuous/vicious cycle that LP-based methods cannot model.

### 2.4 Network Topology

**Discussion:** We implemented two topologies:
- **Base (4-echelon):** Factory → 2 distributors → 2 warehouses → 3 retailers (9 nodes, 11 edges). Tests complex multi-echelon coordination.
- **Serial:** Linear chain with 4 stages. Tests whether Clark-Scarf echelon decomposition provides the theoretical optimum.

**Key finding:** The serial topology is where the echelon base-stock heuristic is theoretically optimal under stationary demand, providing a strong sanity check for the benchmark.

### 2.5 Observation Space Design

**Discussion:** Extensive iteration on what information to provide to neural agents. The final design (V2/V3) includes:
- 8 per-node features: inventory position, lead-time target, gap, on-hand, holding cost, capacity, is_factory, is_retail
- 10 global features: demand velocity, time encoding (normalized + sinusoidal), demand history (5 periods), goodwill sentiment

**Critical decision:** We deliberately excluded future demand from the observation space, creating an "information asymmetry" with the Oracle. This asymmetry is the primary bottleneck for RL performance (accounts for ~15% gap), not architectural limitations.

---

## 3. Agent Development Timeline

### 3.1 Phase 1: Classical OR Baselines

**Oracle (Clairvoyant LP):**
- Implemented as a rolling-horizon LP with perfect future demand knowledge
- Uses PuLP/CBC solver
- Serves as theoretical upper bound (not achievable in practice)
- Special handling for goodwill: iterative solve with demand adjustment

**MSSP (Multi-Stage Stochastic Programming):**
- Scenario tree with depth 3, 10 demand samples
- Both informed (true distribution) and blind (rolling-average estimation) variants
- Near-Oracle in stationary settings (<6% gap)
- Fails under goodwill due to exogenous demand assumption

**DLP (Deterministic Linear Programming):**
- Replaces demand uncertainty with conditional expectations
- 10-period rolling horizon
- ~15× faster than MSSP with competitive performance
- Adaptive horizon variant tested (H = min(max_pipeline_LT + 10, remaining_periods)) — showed modest improvement

**Echelon Base-Stock Heuristic:**
- Clark-Scarf echelon stock policy
- Order-up-to level = CR-quantile of newsvendor distribution
- Both informed and blind variants, backlog and lost-sales modes
- Theoretically optimal for serial networks under stationarity

### 3.2 Phase 2: RL Agents (PPO-MLP)

**RLV1 (Initial Baseline):**
- PPO with 2×256 MLP, 200K timesteps
- Default hyperparameters
- Achieved ~55-70% of Oracle — disappointing initial result

**RLV4 (Tuned Baseline):**
- Extended training (500K+ steps)
- Refined reward normalization (VecNormalize clip=10)
- Improved to ~75-80% of Oracle
- Key learning: reward normalization matters enormously for inventory problems

### 3.3 Phase 3: GNN Agents

**GNN V1 (GCN):**
- Graph Convolutional Network feature extractor
- Symmetric adjacency with Kipf-Welling normalization
- 590K parameters, 80% of Oracle
- First demonstration that graph structure helps

**GNN V2 (MPNN):**
- Directed message passing with edge MLPs
- GRU update for node states
- 603K parameters, but paradoxically performed worse than V1 (79.6%)
- Root cause: feature layout bug (interleaved vs. grouped node features)

**GNN V3 (MPNN + Attention + BatchNorm):**
- Fixed feature layout (grouped by feature type, not by node)
- Added BatchNorm for training stability
- Attention-weighted message aggregation
- 603K parameters, 85.1% of Oracle — best pure RL result
- **Key discussion:** The 5pp improvement from V2→V3 was entirely due to BatchNorm + grouped features, not architectural changes. This highlighted that engineering details matter more than architecture.

### 3.4 Phase 4: Architecture Experiments

**TSPPO (Transformer Sequential PPO):**
- Transformer encoder replacing GNN message passing
- Node tokenization with learned positional + type embeddings
- 3 layers, 4 heads, FFN dim 256
- Result: 82% of Oracle — matched but did not exceed GNN V3

**Discussion on why Transformers didn't help:**
1. Graph structure matters more than global attention for supply chains with fixed topology
2. Temporal resolution too coarse (30 periods, 5 demand history features)
3. The bottleneck is information (no future demand), not model capacity

### 3.5 Phase 5: Imitation Learning

**GNN-IL (Behavioral Cloning + PPO Fine-tuning):**
- 3-phase pipeline: demo collection → BC warm-start → PPO fine-tuning
- 88.9% of Oracle — significant improvement over pure RL
- **Key finding:** PPO fine-tuning after 5K steps actually *degraded* performance, from +$829 to +$670 at 200K. The PPO exploration corrupted the BC policy.

**Specialist DAgger:**
- Iterative imitation learning addressing BC's distribution shift
- 4 specialist models (one per demand type)
- 8 DAgger rounds with linearly decaying mixing parameter
- **94.6% of Oracle** — matching MSSP!
- **Key discussion:** Generalist DAgger collapsed to 75.7%, confirming that the model cannot simultaneously learn optimal policies for conflicting demand patterns. Specialization is critical for IL.

### 3.6 Phase 6: Hybrid Methods

**Residual RL:**
- Heuristic + learned correction: $a_t = a_t^{heur} + \Delta \cdot \delta_t$
- MLP outputs residual in [-1, 1], scaled by max residual (Δ=50)
- 92% of Oracle on stationary, improves in goodwill scenarios
- **Key discussion:** Proportional residuals ($\alpha \in [0, 0.3]$ of heuristic action) were tested but fixed absolute residuals performed better in practice

**Residual RL V2 (Enhanced Features):**
- Used V2 enhanced per-node features
- Domain randomization during training
- Scenario-specific training (stationary, shock × goodwill)
- Results were mixed — domain randomization helped generalization but reduced peak performance on individual scenarios

---

## 4. Benchmark Experiment History

### 4.1 Initial Benchmark (V1/V2 Comparison)

First systematic comparison of GNN V1 vs V2 across 16 scenarios (4 demand × 2 goodwill × 2 backlog). Established the benchmark engine framework.

### 4.2 Unified Benchmark (64 Scenarios)

Expanded to 64 scenarios (4 demand × 2 goodwill × 2 backlog × 2 topology). Split into OR-tier (Oracle, MSSP, DLP, Heuristic) and RL-tier (PPO-MLP, GNN V3, Residual RL) for computational efficiency.

### 4.3 DLP Adaptive Horizon Experiment

**Question:** Does an adaptive planning horizon improve DLP performance?

**Setup:** Compared fixed H=10 with adaptive H=min(max_cumulative_pipeline_LT + 10, remaining_periods).

**Result:** Modest improvement in trend scenarios where the fixed horizon was too short to capture the demand trajectory. Negligible difference in stationary/seasonal. The adaptive horizon never hurt performance, so it was adopted as default.

### 4.4 Combined-Effect Benchmark

Tested demand combinations (trend + seasonal, stationary + shock) to assess agent robustness under compound non-stationarity. GNN showed the most stable degradation, while MSSP's assumptions were more severely violated.

### 4.5 Fair Comparison Experiment

Carefully controlled experiment ensuring all agents see identical demand realizations (same seeds: 1000-1029). This was critical — earlier comparisons used different seeds for different agents, potentially inflating or deflating performance differences.

### 4.6 DAgger Benchmark

Final comprehensive evaluation including DAgger specialists. Used 5 unseen seeds (42, 123, 456, 789, 999) for evaluation. This was the experiment that produced the headline result: DAgger matches MSSP at 94.6% of Oracle.

---

## 5. Paper Writing Process

### 5.1 Abstract Discussion

**Key debate:** How long should the abstract be? Initial drafts were ~250 words, reduced to ~200. The abstract was iteratively refined to:
- Lead with the fragmentation problem (no standard benchmarks)
- Quantify the contribution (10 agents, 32 scenarios)
- Highlight the key finding (no single paradigm dominates)
- Include the practical insight (RL excels under endogenous dynamics)

**Decision on mentioning heuristics:** Initially heuristics were underemphasized. We explicitly added them to the abstract because practitioners care most about simple baselines, and the heuristic's surprisingly strong performance (87-93% of Oracle) is a key finding.

### 5.2 Introduction Structure

Structured around 3 arguments:
1. Inventory management is critical ($1.4T globally)
2. Existing benchmarks are fragmented (different environments, metrics, baselines)
3. Our contribution addresses this gap

### 5.3 Related Work Organization

Organized into 5 subsections: Classical OR, Heuristics, RL, GNNs, Benchmarking Gaps. Each subsection traces the evolution of the field chronologically.

**Key discussion:** Should we highlight emerging SOTA approaches (pretrained time-series for demand planning, LLM-based agents, quantum methods)? Decision: mention them briefly in future work rather than related work, since we didn't evaluate them.

### 5.4 Benchmarking Value Discussion

**User concern:** "Given many solutions' results could differ with any change in network structure or fixed variables, does my benchmarking have value?"

**Resolution:** Yes, precisely because we acknowledge this as a controlled variable. The benchmark's value lies in:
1. **Controlled comparison:** All agents evaluated under identical conditions
2. **Extensibility:** The framework makes it easy to vary each factor systematically
3. **Reproducibility:** Open-source code + fixed seeds = reproducible results
4. **Baseline establishment:** Future works can compare against our published numbers

The limitations section explicitly frames fixed variables (cost structure, lead times, single product, demand distribution, episode length) as controlled experiments rather than weaknesses.

---

## 6. Key Technical Discussions

### 6.1 Information Dominates Architecture

The single most important finding across all experiments: the performance hierarchy is determined by **information access**, not model architecture.

| Information Level | Examples | % Oracle |
|---|---|---|
| Perfect foresight | Oracle | 100% |
| Distribution known | MSSP (Informed) | 95% |
| Expert demos | DAgger Specialist | 94.6% |
| Distribution estimated | MSSP (Blind), DLP | 88-90% |
| Observation only | GNN V3, TSPPO | 82-85% |

GNN V3 and TSPPO achieve *identical* performance despite fundamentally different architectures (message passing vs self-attention), confirming that the ~15% gap to Oracle stems from missing future demand information, not model expressiveness.

### 6.2 The Goodwill Regime Shift

On goodwill-enabled scenarios, the performance ranking *reverses*:
- **Without goodwill:** MSSP > DLP > Heuristic > GNN > MLP
- **With goodwill:** GNN > Residual > Heuristic > MSSP > DLP

This is arguably the paper's most practically relevant finding. Real-world demand often exhibits endogenous features (churn, loyalty, word-of-mouth), and our benchmark shows that classical methods fail precisely in these settings.

### 6.3 Training Stability Concerns

All RL agents exhibited high training variance:
- GNN V3: CV = 18.3% across checkpoints
- TSPPO: CV = 22.1%
- DAgger-Generalist: CV = 148.2% (catastrophic)

This motivated:
1. **Domain randomization:** Randomizing demand pattern each episode
2. **Specialist training:** Separate models per demand type
3. **Checkpoint selection:** Using evaluation performance, not final checkpoint

### 6.4 The Pruning Paradox

GNN V2's underperformance relative to V1 despite having a strictly more expressive architecture. Root cause: interleaved feature layout caused the GNN to conflate different features across nodes. Grouped layout (all nodes' feature_1, then all nodes' feature_2...) resolved this.

### 6.5 Speed-Quality Tradeoff

| Method | Quality (% Oracle) | Speed (per episode) | Speedup vs MSSP |
|--------|-------------------|---------------------|-----------------|
| MSSP | 95% | ~2s | 1× |
| DLP | 90% | ~0.5s | 4× |
| Heuristic | 87% | <1ms | 2000× |
| GNN | 85% | ~15ms | 130× |
| DAgger-S | 94.6% | ~15ms | 130× |

**Key insight for practitioners:** DAgger-S matches MSSP quality at 130× the speed. For real-time applications (e-commerce, automated warehousing), this is the only viable high-quality option.

### 6.6 Multi-Agent Extension

A PettingZoo-based multi-agent, multi-product extension was developed and tested:
- Converted to `ParallelEnv` for simultaneous agent decisions
- Implemented competing markets with demand routing and spillover
- MO-Gymnasium vector rewards for multi-objective optimization
- Validated economic realism of the environment

This extension is documented but not included in the paper's main results, reserved for a follow-up study.

---

## 7. Unresolved Questions & Future Directions

### 7.1 Open Technical Questions

1. **Would longer training close the RL-MSSP gap?** We trained for 200K steps on a laptop. Would 1M+ steps on a GPU change the ranking?
2. **Can a single generalist DAgger model work?** Our generalist collapsed to 75.7%. Would curriculum learning or progressive training help?
3. **What happens with stochastic lead times?** All our experiments use deterministic lead times. This is a significant simplification.
4. **How do results scale with network size?** Our 9-node network is small. Do the relative rankings hold for 50+ node networks?

### 7.2 Promising Research Directions

1. **Foundation models for demand estimation:** TimesFM, Chronos, Moirai as plug-in demand forecasters
2. **LLM-based inventory agents:** Can GPT-4/Claude make reasonable ordering decisions through prompting?
3. **Quantum optimization:** QAOA for the LP relaxation in MSSP
4. **Federated learning:** Multi-supplier settings where demand data cannot be shared
5. **Online leaderboard:** Community-submitted agents evaluated on standardized scenarios

---

*This research log was compiled from 9+ development conversations spanning environment design, agent implementation, benchmark execution, and paper writing. The complete codebase, trained models, and evaluation data are available in this repository.*
