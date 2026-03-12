#!/usr/bin/env python3
"""
collect_oracle_demos.py — Collect Oracle expert demonstrations for GNN-IL.

Runs ClairvoyantOracle across 16 scenarios × 10 seeds, recording:
  - GNN-format observations (DomainFeatureWrapper, enhanced=True, grouped=True)
  - Oracle actions rescaled to [-1, 1] (matching GNN's RescaleAction space)
  - Per-step raw rewards

Output: data/demos/oracle_demos.npz
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import itertools
import numpy as np
from src.envs.core.environment import CoreEnv
from src.envs.wrappers.feature_wrappers import DomainFeatureWrapper
from src.agents.oracle import ClairvoyantOracle

SEEDS = list(range(10))  # 10 seeds: 0-9
SCENARIOS = list(itertools.product(
    ['base'],
    ['stationary', 'trend', 'seasonal', 'shock'],
    [False, True],   # goodwill
    [False, True],   # backlog
))
H = 30  # planning horizon


def collect_demos():
    os.makedirs('data/demos', exist_ok=True)

    all_obs = []
    all_actions = []
    all_rewards = []
    all_scenario_ids = []

    total = len(SCENARIOS) * len(SEEDS)
    count = 0

    for scen_idx, (net, dem, gw, bl) in enumerate(SCENARIOS):
        for seed in SEEDS:
            count += 1
            env_kwargs = {
                'scenario': net,
                'backlog': bl,
                'num_periods': H,
                'demand_config': {'type': dem, 'use_goodwill': gw, 'base_mu': 20},
            }

            # --- Run Oracle to get optimal actions ---
            oracle = ClairvoyantOracle(env_kwargs, planning_horizon=H)
            oracle_actions, oracle_env = oracle.solve(seed=seed)

            # --- Replay with GNN-format wrapper to collect observations ---
            env = CoreEnv(**env_kwargs)
            env = DomainFeatureWrapper(env, is_blind=False, enhanced=True, grouped=True)
            obs, _ = env.reset(seed=seed)

            episode_reward = 0.0
            for t in range(H):
                raw_action = oracle_actions[t]  # [0, 3100]^11

                # Convert to rescaled [-1, 1] for GNN
                act_low = env.unwrapped.action_space.low
                act_high = env.unwrapped.action_space.high
                rescaled_action = 2.0 * (raw_action - act_low) / (act_high - act_low + 1e-8) - 1.0
                rescaled_action = np.clip(rescaled_action, -1.0, 1.0)

                all_obs.append(obs.astype(np.float32))
                all_actions.append(rescaled_action.astype(np.float32))

                # Step with raw action (wrapper expects raw)
                obs, reward, done, trunc, info = env.step(raw_action)
                raw_reward = info.get('raw_reward', reward)
                all_rewards.append(raw_reward)
                all_scenario_ids.append(scen_idx)
                episode_reward += raw_reward

                if done or trunc:
                    break

            if count % 10 == 0 or count == total:
                print(f"  [{count:3d}/{total}] {dem}/G={gw}/B={bl} seed={seed}: Oracle profit={episode_reward:+.1f}", flush=True)

    obs_array = np.stack(all_obs)
    act_array = np.stack(all_actions)
    rew_array = np.array(all_rewards, dtype=np.float32)
    scen_array = np.array(all_scenario_ids, dtype=np.int32)

    np.savez_compressed(
        'data/demos/oracle_demos.npz',
        obs=obs_array,
        actions=act_array,
        rewards=rew_array,
        scenario_ids=scen_array,
    )

    print(f"\n  Saved {obs_array.shape[0]} demo pairs to data/demos/oracle_demos.npz")
    print(f"  Obs shape: {obs_array.shape}, Actions shape: {act_array.shape}")
    print(f"  Scenarios: {len(SCENARIOS)}, Seeds: {len(SEEDS)}, Steps/ep: {H}")


if __name__ == '__main__':
    collect_demos()
