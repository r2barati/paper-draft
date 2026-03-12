# Towards a Standard Benchmark for Inventory Management Methods

[![Build LaTeX PDF](https://github.com/r2barati/paper-draft/actions/workflows/build-pdf.yml/badge.svg)](https://github.com/r2barati/paper-draft/actions/workflows/build-pdf.yml)

An open benchmarking framework for fair, reproducible comparison of inventory management methods across four paradigms: **heuristic policies**, **mathematical programming**, **reinforcement learning**, and **deep learning**.

## Repository Structure

```
├── paper/                           # LaTeX paper source
│   ├── main.tex                     # Main document
│   ├── references.bib               # Bibliography
│   ├── sections/                    # Paper sections
│   │   ├── related_work.tex         # §2 Related Work
│   │   ├── environment.tex          # §3 Benchmarking Environment
│   │   ├── methods.tex              # §4 Solution Methods
│   │   ├── experimental_design.tex  # §5 Experimental Design
│   │   ├── results_comprehensive.tex# §6 Results (DAgger-focused)
│   │   ├── results_analysis.tex     # §6 Results (OR vs RL analysis)
│   │   ├── discussion.tex           # §7 Discussion
│   │   ├── conclusion.tex           # §8 Conclusion & Future Work
│   │   ├── model_comparison.tex     # Agent Architecture Comparison
│   │   └── appendix.tex             # Technical Appendix
│   └── figures/                     # All paper figures (34 PNGs)
│
├── code/                            # Source code
│   ├── src/
│   │   ├── agents/                  # 13 agent implementations
│   │   │   ├── oracle.py            # Clairvoyant Rolling-Horizon LP
│   │   │   ├── mssp_agent.py        # Multi-Stage Stochastic Programming
│   │   │   ├── dlp_agent.py         # Deterministic Linear Programming
│   │   │   ├── heuristic_agent.py   # Echelon Base-Stock
│   │   │   ├── ss_policy_heuristic_agent.py  # (s,S) Reorder Point
│   │   │   ├── ppo_baseline_agent.py  # PPO-MLP (RLV1/RLV4)
│   │   │   ├── ppo_gnn_agent.py       # PPO + GNN Feature Extractor
│   │   │   ├── ppo_residual_agent.py  # Residual RL (Heuristic + PPO)
│   │   │   └── ...                    # Newsvendor, Exp-Smoothing, Baselines
│   │   ├── models/                  # Neural network architectures
│   │   │   ├── gnn_extractor.py     # GNN V1 (GCN)
│   │   │   ├── gnn_extractor_v2.py  # GNN V2 (MPNN, directed edges)
│   │   │   ├── gnn_extractor_v3.py  # GNN V3 (MPNN + attention + BN)
│   │   │   ├── transformer_extractor.py  # TSPPO (Transformer encoder)
│   │   │   └── shared_mlp_extractor.py   # Shared MLP actor-critic
│   │   └── envs/                    # Gymnasium environment
│   │       ├── builder.py           # Scenario builder
│   │       ├── core/                # Environment, demand, topology
│   │       └── wrappers/            # Feature, action, logging wrappers
│   ├── scripts/
│   │   ├── train/                   # 16 training scripts
│   │   └── eval/                    # 16 benchmark scripts
│   └── tests/                       # Unit tests
│
├── data/
│   ├── results/                     # 19 CSV benchmark results
│   └── models/                      # 57 trained model checkpoints
│
├── docs/                            # Documentation
│   ├── README_AGENTS.md             # Agent implementation guide
│   ├── README_DATA.md               # Data format documentation
│   ├── README_ENVIRONMENT.md        # Environment configuration guide
│   ├── project_overview.md          # Full project overview
│   ├── scientific_peer_review.md    # Self-review for paper quality
│   └── audit_report.md              # Code audit report
│
├── RESEARCH_LOG.md                  # Comprehensive research log
├── README.md                        # This file
└── .github/workflows/build-pdf.yml  # CI: LaTeX → PDF on push
```

## Solution Methods

| Agent | Paradigm | Info Access | Training | % Oracle |
|-------|----------|-------------|----------|----------|
| Oracle | LP Upper Bound | Perfect foresight | None | 100% |
| MSSP | Math Programming | Distribution known | None | 95% |
| DAgger-S | Imitation Learning | Observed | DAgger | **94.6%** |
| DLP | Math Programming | Distribution known | None | 90% |
| GNN-IL | Imitation Learning | Observed | BC + PPO | 88.9% |
| Heuristic | Echelon Base-Stock | Estimated | None | 87% |
| GNN V3 | RL + GNN | Observed | PPO | 85.1% |
| TSPPO | RL + Transformer | Observed | PPO | 82.0% |
| Residual RL | Hybrid | Observed | PPO | 82.6% |
| PPO-MLP | RL | Observed | PPO | ~75% |

## Key Findings

1. **No single paradigm dominates.** MSSP excels under stationary demand (<6% gap); GNN-based RL excels under endogenous goodwill dynamics (25% improvement over MSSP).
2. **Information dominates architecture.** GNN V3 and TSPPO achieve identical performance despite fundamentally different architectures, confirming the bottleneck is information, not capacity.
3. **Specialist DAgger matches MSSP** at 94.6% of Oracle while running **130× faster**.
4. **RL agents uniquely handle endogenous dynamics.** All RL/IL agents surpass the Oracle by 5–15% in goodwill scenarios.
5. **The speed–quality tradeoff is dramatic.** RL agents are 100–300× faster than MSSP at inference.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/r2barati/paper-draft.git
cd paper-draft

# Run a benchmark
python code/scripts/eval/benchmark.py

# Train a GNN agent
python code/scripts/train/train_ppo_gnn_v3.py
```

## Building the Paper

The CI automatically compiles `paper/main.tex` to PDF on every push. Download the PDF from the [Actions tab](https://github.com/r2barati/paper-draft/actions).

## Citation

```bibtex
@article{barati2025benchmark,
  title  = {Towards a Standard Benchmark for Inventory Management Methods},
  author = {Barati, Reza},
  year   = {2025}
}
```
