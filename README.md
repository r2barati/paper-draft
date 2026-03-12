# Benchmark for Inventory Management Methods

This repository contains the source code, paper sections, and appendix materials for the paper **"Towards a Standard Benchmark for Inventory Management Methods"**.

## Repository Structure

```
├── main.tex                         # Main LaTeX document
├── references.bib                   # Bibliography
├── figures/                         # Paper figures
│   ├── method_taxonomy.png          # Solution method paradigms
│   ├── literature_timeline.png      # Research timeline
│   ├── E_Network_Topology_Mapping.png
│   └── ...                          # Other charts
│
├── section2_related_work.tex        # Section 2: Related Work
├── section3_environment.tex         # Section 3: Benchmarking Environment
├── section4_methods.tex             # Section 4: Solution Methods
├── section5_experimental_design.tex # Section 5: Experimental Design
├── section5_results.tex             # Comprehensive Results (DAgger)
├── section6_results.tex             # Section 6: Results
├── section7_discussion.tex          # Section 7: Discussion
├── section8_conclusion.tex          # Section 8: Conclusion & Future Work
├── section_model_comparison.tex     # Agent Architecture Comparison
├── appendix_technical.tex           # Full Technical Appendix
│
├── code/                            # Source code
│   ├── src/
│   │   ├── agents/                  # Agent implementations
│   │   │   ├── oracle.py            # Clairvoyant Rolling-Horizon Oracle (LP)
│   │   │   ├── mssp_agent.py        # Multi-Stage Stochastic Programming
│   │   │   ├── dlp_agent.py         # Deterministic Linear Programming
│   │   │   ├── heuristic_agent.py   # Echelon Base-Stock Heuristic
│   │   │   ├── ss_policy_heuristic_agent.py  # (s,S) Reorder-Point Policy
│   │   │   ├── newsvendor_heuristic_agent.py # Newsvendor Critical Ratio
│   │   │   ├── exp_smoothing_heuristic_agent.py # Exponential Smoothing
│   │   │   ├── ppo_baseline_agent.py  # PPO-MLP Baseline (RLV1/RLV4)
│   │   │   ├── ppo_gnn_agent.py       # PPO with GNN Feature Extractor
│   │   │   ├── ppo_residual_agent.py  # Residual RL (Heuristic + PPO)
│   │   │   └── baselines.py          # Dummy Agent
│   │   │
│   │   ├── models/                  # Neural network architectures
│   │   │   ├── gnn_extractor.py     # GNN V1 (GCN)
│   │   │   ├── gnn_extractor_v2.py  # GNN V2 (MPNN, directed edges)
│   │   │   ├── gnn_extractor_v3.py  # GNN V3 (MPNN + attention + BatchNorm)
│   │   │   ├── transformer_extractor.py # TSPPO (Transformer encoder)
│   │   │   └── shared_mlp_extractor.py  # Shared MLP for actor-critic
│   │   │
│   │   └── envs/                    # Environment
│   │       ├── builder.py           # Scenario builder (topology + demand)
│   │       ├── core/
│   │       │   ├── environment.py   # Gymnasium environment implementation
│   │       │   ├── demand_engine.py # Composable demand patterns
│   │       │   └── network_topology.py # DAG topology configs
│   │       └── wrappers/
│   │           ├── feature_wrappers.py  # DomainFeatureWrapper (V2/V3 obs)
│   │           ├── action_wrappers.py   # ResidualActionWrapper, RescaleAction
│   │           ├── logging_wrappers.py  # Episode logging
│   │           └── per_link_wrapper.py  # Per-link observation wrapper
│   │
│   ├── scripts/                     # Training and evaluation scripts
│   │   ├── train/                   # Training scripts for all agents
│   │   └── eval/                    # Benchmark evaluation scripts
│   │
│   └── tests/                       # Test files
│
└── .github/
    └── workflows/
        └── build-pdf.yml            # CI: LaTeX → PDF on push
```

## Solution Methods (10 agents across 4 paradigms)

| Agent | Paradigm | Info Access | Training |
|-------|----------|-------------|----------|
| Oracle | LP Upper Bound | Perfect foresight | None |
| MSSP (Informed/Blind) | Math Programming | True/Estimated | None |
| DLP (Informed/Blind) | Math Programming | True/Estimated | None |
| Echelon Base-Stock | Heuristic | True/Estimated | None |
| (s,S) Reorder Point | Heuristic | True/Estimated | None |
| PPO-MLP (RLV1/RLV4) | RL | Observed | PPO |
| PPO-GNN (V1/V2/V3) | RL + GNN | Observed | PPO |
| Residual RL | Hybrid | Observed | PPO |
| GNN-IL | Imitation Learning | Observed | BC + PPO |
| Specialist DAgger | Imitation Learning | Observed | DAgger |
| TSPPO | RL + Transformer | Observed | PPO |

## Key Results

- **No single paradigm dominates universally**: MSSP excels under stationary demand (<6% gap), while GNN-based RL excels under endogenous dynamics (25% improvement over MSSP).
- **Specialist DAgger matches MSSP** at 94.6% of Oracle while running 130× faster.
- **RL agents uniquely handle endogenous dynamics**: All RL/IL agents surpass the Oracle by 5–15% in goodwill scenarios.
- **RL inference is 100–300× faster** than MSSP, enabling real-time applications.

## Building the Paper

The GitHub Action automatically compiles `main.tex` to PDF on every push. The PDF artifact is available in the Actions tab.

## Citation

```bibtex
@article{barati2025benchmark,
  title  = {Towards a Standard Benchmark for Inventory Management Methods},
  author = {Barati, Reza},
  year   = {2025}
}
```

## License

This project is released for academic use. Please cite the paper if you use this code.
