#!/usr/bin/env python3
"""
benchmark_gnn_dagger.py — Full 16-scenario benchmark of GNN-DAgger (best checkpoint)
vs V3, GNN-IL, Oracle on unseen seeds.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pickle
import itertools
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import RescaleAction

from src.envs.core.environment import CoreEnv
from src.envs.wrappers.feature_wrappers import DomainFeatureWrapper
from src.agents.oracle import ClairvoyantOracle
from src.models.gnn_extractor_v3 import GNNFeaturesExtractorV3

UNSEEN_SEEDS = [42, 123, 456, 789, 999]
SCENARIOS = list(itertools.product(
    ['stationary', 'trend', 'seasonal', 'shock'],
    [False, True], [False, True],
))
H = 30


def eval_model(env_kwargs, seed, model, vec_norm):
    env = CoreEnv(**env_kwargs)
    env = DomainFeatureWrapper(env, is_blind=False, enhanced=True, grouped=True)
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)
    obs, _ = env.reset(seed=seed)
    ep_r = 0
    for t in range(H):
        obs_n = vec_norm.normalize_obs(obs.reshape(1, -1)).squeeze(0)
        action, _ = model.predict(obs_n, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        ep_r += info.get('raw_reward', reward)
        if done or trunc: break
    return ep_r


def eval_oracle(env_kwargs, seed):
    oracle = ClairvoyantOracle(env_kwargs, planning_horizon=H)
    actions, _ = oracle.solve(seed=seed)
    env = CoreEnv(**env_kwargs)
    env.reset(seed=seed)
    profit = 0
    for t in range(H):
        _, reward, done, trunc, _ = env.step(actions[t])
        profit += reward
        if done or trunc: break
    return profit


def load_model(path, vecnorm_path):
    model = PPO.load(path, custom_objects={'GNNFeaturesExtractorV3': GNNFeaturesExtractorV3})
    with open(vecnorm_path, 'rb') as f:
        vn = pickle.load(f)
    vn.training = False
    return model, vn


def main():
    print("=" * 80)
    print("  COMPREHENSIVE BENCHMARK: DAgger vs V3 vs GNN-IL vs Oracle")
    print(f"  Unseen seeds: {UNSEEN_SEEDS}")
    print("=" * 80)

    # Load models
    print("Loading models...", flush=True)
    dag_m, dag_vn = load_model('data/models/ppo_gnn_dagger_best', 'data/models/vec_normalize_gnn_dagger.pkl')
    v3_m, v3_vn = load_model('data/models/ppo_gnn_v3', 'data/models/vec_normalize_gnn_v3.pkl')
    il_m, il_vn = load_model('gnn_il_logs/best_model', 'data/models/vec_normalize_gnn_il.pkl')

    print("  DAgger ✓  V3 ✓  GNN-IL ✓\n")

    results = []
    for dem, gw, bl in SCENARIOS:
        env_kwargs = {'scenario': 'base', 'backlog': bl, 'num_periods': H,
                      'demand_config': {'type': dem, 'use_goodwill': gw, 'base_mu': 20}}

        oracle_ps, v3_ps, il_ps, dag_ps = [], [], [], []
        for seed in UNSEEN_SEEDS:
            oracle_ps.append(eval_oracle(env_kwargs, seed))
            v3_ps.append(eval_model(env_kwargs, seed, v3_m, v3_vn))
            il_ps.append(eval_model(env_kwargs, seed, il_m, il_vn))
            dag_ps.append(eval_model(env_kwargs, seed, dag_m, dag_vn))

        o, v3, il, dag = np.mean(oracle_ps), np.mean(v3_ps), np.mean(il_ps), np.mean(dag_ps)
        v3p = v3/o*100 if o > 0 else 0
        ilp = il/o*100 if o > 0 else 0
        dagp = dag/o*100 if o > 0 else 0

        results.append({'demand': dem, 'goodwill': gw, 'backlog': bl,
                        'oracle': o, 'v3': v3, 'il': il, 'dagger': dag,
                        'v3_pct': v3p, 'il_pct': ilp, 'dag_pct': dagp})

        best_tag = ""
        best_val = max(v3, il, dag)
        if best_val == dag: best_tag = " ★"
        elif best_val == il: best_tag = " ◆"

        print(f"  {dem:12s} G={str(gw):5s} B={str(bl):5s}: "
              f"Oracle={o:+8.1f}  V3={v3:+8.1f}({v3p:5.1f}%)  "
              f"IL={il:+8.1f}({ilp:5.1f}%)  "
              f"DAgger={dag:+8.1f}({dagp:5.1f}%){best_tag}", flush=True)

    # Summary
    print("\n" + "=" * 80)
    fair = [r for r in results if not r['goodwill']]
    gw = [r for r in results if r['goodwill']]

    print("  NO-GOODWILL (fair, 8 scenarios):")
    print(f"    V3:     {np.mean([r['v3_pct'] for r in fair]):5.1f}% of Oracle")
    print(f"    GNN-IL: {np.mean([r['il_pct'] for r in fair]):5.1f}% of Oracle")
    print(f"    DAgger: {np.mean([r['dag_pct'] for r in fair]):5.1f}% of Oracle")

    print("  GOODWILL (8 scenarios):")
    print(f"    V3:     {np.mean([r['v3_pct'] for r in gw]):5.1f}% of Oracle")
    print(f"    GNN-IL: {np.mean([r['il_pct'] for r in gw]):5.1f}% of Oracle")
    print(f"    DAgger: {np.mean([r['dag_pct'] for r in gw]):5.1f}% of Oracle")

    dag_wins_v3 = sum(1 for r in results if r['dagger'] > r['v3'])
    dag_wins_il = sum(1 for r in results if r['dagger'] > r['il'])
    print(f"\n  DAgger beats V3:     {dag_wins_v3}/16")
    print(f"  DAgger beats GNN-IL: {dag_wins_il}/16")

    import json
    with open('data/results/gnn_dagger_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: data/results/gnn_dagger_benchmark.json")


if __name__ == '__main__':
    main()
