import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import itertools
import time
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from src.envs.core.environment import CoreEnv
from src.agents.oracle import StandaloneOracleOptimizer
from src.agents.dlp_agent import RollingHorizonDLPAgent
from src.agents.mssp_agent import RollingHorizonMSSPAgent
from src.agents.heuristic_agent import HeuristicAgent
from src.agents.ss_policy_heuristic_agent import SSPolicyHeuristicAgent
from src.agents.exp_smoothing_heuristic_agent import ExpSmoothingHeuristicAgent

def _kpi(env, episode_reward):
    avg_inv = np.mean([np.sum(env.X[t]) for t in range(env.period)]) if env.period > 0 else 0
    total_u = np.sum(env.U)
    total_d = np.sum(env.D)
    fill_rate = max(0.0, 1.0 - (total_u / total_d)) if total_d > 0 else 1.0
    return {"profit": episode_reward, "avg_inv": avg_inv,
            "unfulfilled": total_u, "fill_rate": fill_rate}

def _run_oracle(env_kwargs, seed, planning_horizon):
    if env_kwargs.get('demand_config', {}).get('use_goodwill', False) == True:
        from src.agents.baselines import OptimisticEndogenousOracle
        oracle = OptimisticEndogenousOracle(env_kwargs, planning_horizon=planning_horizon)
        optimal_actions = oracle.solve(seed)
        if optimal_actions is None:
            return None
            
        env = CoreEnv(**env_kwargs)
        env.reset(seed=seed)
        episode_reward = 0
        for t in range(planning_horizon):
            _, reward, _, _, _ = env.step(optimal_actions[t])
            episode_reward += reward
        return _kpi(env, episode_reward)
    else:
        probe_env = CoreEnv(**env_kwargs)
        probe_env.reset(seed=seed)
        for _ in range(planning_horizon):
            probe_env.step(np.zeros(probe_env.action_space.shape))
        demand_trace = {}
        for i, edge in enumerate(probe_env.network.retail_links):
            demand_trace[edge] = probe_env.D[:, i].copy()

        fixed_cfg = env_kwargs.copy()
        fixed_cfg['user_D'] = demand_trace
        env = CoreEnv(**fixed_cfg)
        env.reset(seed=seed)
        oracle = StandaloneOracleOptimizer(
            env, known_demand_scenario=demand_trace,
            planning_horizon=planning_horizon, is_continuous=True)
        optimal_actions = oracle.solve_full_horizon()
        if optimal_actions is None:
            return None
        episode_reward = 0
        for t in range(planning_horizon):
            _, reward, _, _, _ = env.step(optimal_actions[t])
            episode_reward += reward
        return _kpi(env, episode_reward)

def _run_dlp(env_kwargs, seed, planning_horizon, is_blind=False):
    env = CoreEnv(**env_kwargs)
    obs, _ = env.reset(seed=seed)
    agent = RollingHorizonDLPAgent(env, planning_horizon=10, is_blind=is_blind)
    episode_reward = 0
    for t in range(planning_horizon):
        obs, reward, terminated, truncated, _ = env.step(agent.get_action(t))
        episode_reward += reward
        if terminated or truncated: break
    return _kpi(env, episode_reward)

def _run_mssp(env_kwargs, seed, planning_horizon, is_blind=False):
    env = CoreEnv(**env_kwargs)
    obs, _ = env.reset(seed=seed)
    agent = RollingHorizonMSSPAgent(env, planning_horizon=10, branching_depth=3, is_blind=is_blind)
    episode_reward = 0
    for t in range(planning_horizon):
        obs, reward, terminated, truncated, _ = env.step(agent.get_action(t))
        episode_reward += reward
        if terminated or truncated: break
    return _kpi(env, episode_reward)

def _run_heuristic(env_kwargs, seed, planning_horizon, is_blind=False):
    env = CoreEnv(**env_kwargs)
    obs, _ = env.reset(seed=seed)
    agent = HeuristicAgent(env, is_blind=is_blind)
    episode_reward = 0
    for t in range(planning_horizon):
        obs, reward, terminated, truncated, _ = env.step(agent.get_action(obs, t))
        episode_reward += reward
        if terminated or truncated: break
    return _kpi(env, episode_reward)

def _run_ss_policy(env_kwargs, seed, planning_horizon, is_blind=False):
    env = CoreEnv(**env_kwargs)
    obs, _ = env.reset(seed=seed)
    agent = SSPolicyHeuristicAgent(env, is_blind=is_blind)
    episode_reward = 0
    for t in range(planning_horizon):
        obs, reward, terminated, truncated, _ = env.step(agent.get_action(obs, t))
        episode_reward += reward
        if terminated or truncated: break
    return _kpi(env, episode_reward)

def _run_exp_smoothing(env_kwargs, seed, planning_horizon, is_blind=False):
    env = CoreEnv(**env_kwargs)
    obs, _ = env.reset(seed=seed)
    agent = ExpSmoothingHeuristicAgent(env, is_blind=is_blind)
    episode_reward = 0
    for t in range(planning_horizon):
        obs, reward, terminated, truncated, _ = env.step(agent.get_action(obs, t))
        episode_reward += reward
        if terminated or truncated: break
    return _kpi(env, episode_reward)

def _run_dummy(env_kwargs, seed, planning_horizon):
    env = CoreEnv(**env_kwargs)
    obs, _ = env.reset(seed=seed)
    episode_reward = 0
    for _ in range(planning_horizon):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        episode_reward += reward
        if terminated or truncated: break
    return _kpi(env, episode_reward)

# --- RL Helper Functions ---

def _run_rl_v1_style(env_kwargs, seed, planning_horizon, model_path, stats_path):
    import os as _os
    zip_path = model_path if model_path.endswith('.zip') else model_path + '.zip'
    if not _os.path.exists(zip_path): return None
    from rl_agent import RLAgent
    env = CoreEnv(**env_kwargs)
    obs, _ = env.reset(seed=seed)
    agent = RLAgent(env, model_path, stats_path, verbose=False)
    episode_reward = 0
    for t in range(planning_horizon):
        action = agent.get_action(obs, t)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        if terminated or truncated: break
    return _kpi(env, episode_reward)

def _run_rl_v2_style(env_kwargs, seed, planning_horizon, model_path, stats_path):
    import os as _os
    zip_path = model_path if model_path.endswith('.zip') else model_path + '.zip'
    if not _os.path.exists(zip_path): return None
    from rl_agent_v2 import RLAgentV2
    env = CoreEnv(**env_kwargs)
    obs, _ = env.reset(seed=seed)
    agent = RLAgentV2(env, model_path, stats_path, deterministic=True)
    episode_reward = 0
    for t in range(planning_horizon):
        action = agent.get_action(obs, t)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        if terminated or truncated: break
    return _kpi(env, episode_reward)

def _run_rl_gnn_style(env_kwargs, seed, planning_horizon, model_path, stats_path):
    import os as _os
    import pickle
    from stable_baselines3 import PPO
    from src.envs.wrappers.feature_wrappers import DomainFeatureWrapper
    from gymnasium.wrappers import RescaleAction
    from src.models.gnn_extractor import GNNFeaturesExtractor

    zip_path = model_path if model_path.endswith('.zip') else model_path + '.zip'
    if not _os.path.exists(zip_path): return None

    env = CoreEnv(**env_kwargs)
    env = DomainFeatureWrapper(env, is_blind=False)
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)
    
    custom_objects = {"GNNFeaturesExtractor": GNNFeaturesExtractor}
    model = PPO.load(model_path, custom_objects=custom_objects)
    
    with open(stats_path, 'rb') as f:
        norm_env = pickle.load(f)
    norm_env.training = False
    norm_env.norm_reward = False

    obs, _ = env.reset(seed=seed)
    episode_reward = 0
    for t in range(planning_horizon):
        obs_2d = obs.reshape(1, -1)
        norm_obs = norm_env.normalize_obs(obs_2d).squeeze(0)
        action, _ = model.predict(norm_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += info.get('raw_reward', reward)
        if terminated or truncated: break
        
    return _kpi(env.unwrapped, episode_reward)

def _run_rl_v4_style(env_kwargs, seed, planning_horizon, model_path, stats_path):
    import os as _os
    import pickle
    zip_path = model_path if model_path.endswith('.zip') else model_path + '.zip'
    if not _os.path.exists(zip_path): return None
    from stable_baselines3 import PPO
    from src.envs.wrappers.feature_wrappers import DomainFeatureWrapper
    from gymnasium.wrappers import RescaleAction

    env = CoreEnv(**env_kwargs)
    env = DomainFeatureWrapper(env, is_blind=False)
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)
    
    model = PPO.load(model_path)
    with open(stats_path, 'rb') as f:
        norm_env = pickle.load(f)
    norm_env.training = False
    norm_env.norm_reward = False

    obs, _ = env.reset(seed=seed)
    episode_reward = 0
    for t in range(planning_horizon):
        obs_2d = obs.reshape(1, -1)
        norm_obs = norm_env.normalize_obs(obs_2d).squeeze(0)
        action, _ = model.predict(norm_obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += info.get('raw_reward', reward)
        if terminated or truncated: break
        
    return _kpi(env.unwrapped, episode_reward)

def _run_rl_residual_style(env_kwargs, seed, planning_horizon, model_path, stats_path):
    import os as _os
    import pickle
    zip_path = model_path if model_path.endswith('.zip') else model_path + '.zip'
    if not _os.path.exists(zip_path): return None
    from stable_baselines3 import PPO
    from src.envs.builder import make_supply_chain_env
    from src.agents.heuristic_agent import HeuristicAgent as _Heuristic

    env_cfg = {
        'scenario': env_kwargs['scenario'],
        'demand_config': env_kwargs['demand_config'],
        'num_periods': env_kwargs['num_periods'],
        'backlog': env_kwargs['backlog']
    }
    
    dummy_env = make_supply_chain_env(agent_type='or', **env_cfg)
    heuristic = _Heuristic(dummy_env.unwrapped, is_blind=False)
    env = make_supply_chain_env(agent_type='residual_rl', heuristic_agent=heuristic, max_residual=50.0, **env_cfg)
    
    model = PPO.load(model_path)
    with open(stats_path, 'rb') as f:
        norm_env = pickle.load(f)
    norm_env.training = False
    norm_env.norm_reward = False

    obs, _ = env.reset(seed=seed)
    episode_reward = 0
    for t in range(planning_horizon):
        obs_2d = obs.reshape(1, -1)
        norm_obs = norm_env.normalize_obs(obs_2d).squeeze(0)
        action, _ = model.predict(norm_obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        if terminated or truncated: break
        
    return _kpi(env.unwrapped, episode_reward)

# Expandable list of RL models to evaluate
# You can add new versions of RL agents to this list without altering core logic
RL_MODELS = [
    {
        'id': 'Residual',
        'model_path': 'data/models/ppo_residual',
        'stats_path': 'data/models/vec_normalize_residual.pkl',
        'runner': _run_rl_residual_style
    },
    {
        'id': 'RLGNN',
        'model_path': 'data/models/ppo_gnn',
        'stats_path': 'data/models/vec_normalize_gnn.pkl',
        'runner': _run_rl_gnn_style
    }
]

def _acc(metrics, res):
    if res is not None:
        for k, v in res.items(): metrics[k].append(v)
def _mean(lst): return np.mean(lst) if lst else np.nan
def _std(lst): return np.std(lst) if lst else np.nan

# --- Evaluation Blocks ---

def _evaluate_baselines(scenario, max_episodes, planning_horizon):
    # Build demand config — support composable effects via DEMAND_CONFIGS lookup
    from scripts.eval.benchmark_engine.config import DEMAND_CONFIGS
    demand_type = scenario['demand_type']
    base_cfg = DEMAND_CONFIGS.get(demand_type, {'type': demand_type, 'base_mu': 20})
    demand_config = {**base_cfg, 'use_goodwill': scenario['use_goodwill']}

    env_kwargs = {
        'scenario': scenario['network'],
        'backlog': scenario['backlog'],
        'demand_config': demand_config,
        'num_periods': planning_horizon
    }

    metrics = {agent: {'profit': [], 'fill_rate': [], 'avg_inv': [], 'unfulfilled': []}
               for agent in ['Oracle', 'MSSP', 'MSSP_Blind', 'DLP', 'DLP_Blind', 'Dummy',
                             'Heuristic', 'Heuristic_Blind', 'SS_Policy', 'SS_Policy_Blind',
                             'ExpSmoothing', 'ExpSmoothing_Blind']}
    
    t_agent = {agent: [] for agent in metrics.keys()}
    scenario_start = time.perf_counter()

    for episode in range(max_episodes):
        seed = 42 + episode
        
        def track(a_name, func, *args, **kwargs):
            t0 = time.perf_counter()
            _acc(metrics[a_name], func(*args, **kwargs))
            t_agent[a_name].append(time.perf_counter() - t0)

        track('Oracle', _run_oracle, env_kwargs, seed, planning_horizon)
        track('MSSP', _run_mssp, env_kwargs, seed, planning_horizon, is_blind=False)
        track('MSSP_Blind', _run_mssp, env_kwargs, seed, planning_horizon, is_blind=True)
        track('DLP', _run_dlp, env_kwargs, seed, planning_horizon, is_blind=False)
        track('DLP_Blind', _run_dlp, env_kwargs, seed, planning_horizon, is_blind=True)
        track('Dummy', _run_dummy, env_kwargs, seed, planning_horizon)
        track('Heuristic', _run_heuristic, env_kwargs, seed, planning_horizon, is_blind=False)
        track('Heuristic_Blind', _run_heuristic, env_kwargs, seed, planning_horizon, is_blind=True)
        track('SS_Policy', _run_ss_policy, env_kwargs, seed, planning_horizon, is_blind=False)
        track('SS_Policy_Blind', _run_ss_policy, env_kwargs, seed, planning_horizon, is_blind=True)
        track('ExpSmoothing', _run_exp_smoothing, env_kwargs, seed, planning_horizon, is_blind=False)
        track('ExpSmoothing_Blind', _run_exp_smoothing, env_kwargs, seed, planning_horizon, is_blind=True)

    elapsed = time.perf_counter() - scenario_start
    
    res = {
        'Network': scenario['network'],
        'Demand': scenario['demand_type'],
        'Goodwill': scenario['use_goodwill'],
        'Backlog': scenario['backlog']
    }
    
    for agent, m in metrics.items():
        res[f'{agent}_Profit'] = _mean(m['profit'])
        res[f'{agent}_FillRate'] = _mean(m['fill_rate'])
        # Intentionally record avg_inv and unfulfilled for all agents
        res[f'{agent}_AvgInv'] = _mean(m['avg_inv'])
        res[f'{agent}_Unfulfilled'] = _mean(m['unfulfilled'])

    res['Oracle_Profit_Std'] = _std(metrics['Oracle']['profit'])
    res['MSSP_Profit_Std'] = _std(metrics['MSSP']['profit'])
    res['DLP_Profit_Std'] = _std(metrics['DLP']['profit'])
    
    res['VPI'] = _mean(metrics['Oracle']['profit']) - _mean(metrics['MSSP']['profit'])
    res['VSS'] = _mean(metrics['MSSP']['profit']) - _mean(metrics['DLP']['profit'])
    res['VPF_MSSP'] = _mean(metrics['MSSP']['profit']) - _mean(metrics['MSSP_Blind']['profit'])
    res['VPF_DLP'] = _mean(metrics['DLP']['profit']) - _mean(metrics['DLP_Blind']['profit'])
    res['VPF_Heuristic'] = _mean(metrics['Heuristic']['profit']) - _mean(metrics['Heuristic_Blind']['profit'])
    res['VPF_SS_Policy'] = _mean(metrics['SS_Policy']['profit']) - _mean(metrics['SS_Policy_Blind']['profit'])
    res['VPF_ExpSmoothing'] = _mean(metrics['ExpSmoothing']['profit']) - _mean(metrics['ExpSmoothing_Blind']['profit'])

    if len(metrics['Oracle']['profit']) > 1 and len(metrics['MSSP']['profit']) > 1:
        res['VPI_pval'] = ttest_rel(metrics['Oracle']['profit'], metrics['MSSP']['profit'])[1]
    else: res['VPI_pval'] = np.nan

    if len(metrics['MSSP']['profit']) > 1 and len(metrics['DLP']['profit']) > 1:
        res['VSS_pval'] = ttest_rel(metrics['MSSP']['profit'], metrics['DLP']['profit'])[1]
    else: res['VSS_pval'] = np.nan

    for agent, t in t_agent.items():
        res[f'Time_{agent}'] = np.mean(t) if t else np.nan
    res['Time_Sec_Baselines'] = elapsed

    return res

def _evaluate_rl_model(scenario, rl_cfg, max_episodes, planning_horizon):
    # Build demand config — support composable effects via DEMAND_CONFIGS lookup
    from scripts.eval.benchmark_engine.config import DEMAND_CONFIGS
    demand_type = scenario['demand_type']
    base_cfg = DEMAND_CONFIGS.get(demand_type, {'type': demand_type, 'base_mu': 20})
    demand_config = {**base_cfg, 'use_goodwill': scenario['use_goodwill']}

    env_kwargs = {
        'scenario': scenario['network'],
        'backlog': scenario['backlog'],
        'demand_config': demand_config,
        'num_periods': planning_horizon
    }

    metrics = {'profit': [], 'fill_rate': [], 'avg_inv': [], 'unfulfilled': []}
    t_episodes = []
    
    scenario_start = time.perf_counter()
    for episode in range(max_episodes):
        seed = 42 + episode
        t0 = time.perf_counter()
        
        try:
            stats = rl_cfg['runner'](env_kwargs, seed, planning_horizon, rl_cfg['model_path'], rl_cfg['stats_path'])
            _acc(metrics, stats)
        except Exception as e:
            if episode == 0:
                print(f"  [Skipped - {type(e).__name__}: {str(e)}]")
            # If the model fails on the first episode, it will fail on all due to shape mismatches
            break
            
        t_episodes.append(time.perf_counter() - t0)

    elapsed = time.perf_counter() - scenario_start
    
    prefix = rl_cfg['id']
    res = {
        'Network': scenario['network'],
        'Demand': scenario['demand_type'],
        'Goodwill': scenario['use_goodwill'],
        'Backlog': scenario['backlog'],
        f'{prefix}_Profit': _mean(metrics['profit']),
        f'{prefix}_FillRate': _mean(metrics['fill_rate']),
        f'{prefix}_AvgInv': _mean(metrics['avg_inv']),
        f'{prefix}_Unfulfilled': _mean(metrics['unfulfilled']),
        f'{prefix}_Profit_Std': _std(metrics['profit']),
        f'Time_{prefix}': np.mean(t_episodes) if t_episodes else np.nan,
        f'Time_Sec_{prefix}': elapsed
    }
    return res

class BenchmarkSuite:
    def __init__(self, max_episodes=5, planning_horizon=30):
        self.max_episodes = max_episodes
        self.planning_horizon = planning_horizon
        self.networks = ['base', 'serial']
        self.demand_types = [
            'stationary', 'trend', 'seasonal', 'shock',
            'trend+seasonal', 'trend+shock', 'seasonal+shock', 'trend+seasonal+shock',
        ]
        self.goodwills = [False, True]
        self.backlogs = [True, False]

    def generate_scenarios(self):
        keys = ['network', 'demand_type', 'use_goodwill', 'backlog']
        values = [self.networks, self.demand_types, self.goodwills, self.backlogs]
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def _get_missing_scenarios(self, all_scenarios, csv_file):
        """Returns the sub-list of scenarios that do NOT already exist in the csv."""
        if not os.path.exists(csv_file):
            return all_scenarios
        try:
            df = pd.read_csv(csv_file)
            completed = set()
            for _, row in df.iterrows():
                completed.add((row['Network'], row['Demand'], bool(row['Goodwill']), bool(row['Backlog'])))
            return [s for s in all_scenarios if (s['network'], s['demand_type'], s['use_goodwill'], s['backlog']) not in completed]
        except Exception:
            return all_scenarios

    def run(self):
        all_scenarios = self.generate_scenarios()
        print("=" * 80)
        print(f"Iterative Benchmark Suite initialized.")
        print(f"Total scenarios in matrix: {len(all_scenarios)}")
        print(f"Episodes per run: {self.max_episodes}")
        print("=" * 80)

        # 1. Run Baselines
        baseline_csv = 'data/results/cache/benchmark_baselines_cache.csv'
        missing_baselines = self._get_missing_scenarios(all_scenarios, baseline_csv)
        if len(missing_baselines) == 0:
            print("[1/2] All baselines already completed.")
        else:
            print(f"[1/2] Running {len(missing_baselines)} missing baselines...")
            for i, scen in enumerate(missing_baselines):
                print(f"  [{i+1}/{len(missing_baselines)}] {scen['network']:6s} | {scen['demand_type']:10s} | gw={scen['use_goodwill']} | bl={scen['backlog']}...", end="", flush=True)
                res = _evaluate_baselines(scen, self.max_episodes, self.planning_horizon)
                df_res = pd.DataFrame([res])
                df_res.to_csv(baseline_csv, mode='a', header=not os.path.exists(baseline_csv), index=False)
                print(f" DONE. Oracle={res['Oracle_Profit']:.1f}, MSSP={res['MSSP_Profit']:.1f}")

        # 2. Run RL Models dynamically
        for rconfig in RL_MODELS:
            prefix = rconfig['id']
            rl_csv = f'data/results/cache/benchmark_{prefix}_cache.csv'
            missing_rl = self._get_missing_scenarios(all_scenarios, rl_csv)
            if len(missing_rl) == 0:
                print(f"\n[2/2] {prefix}: All scenarios completed.")
            else:
                print(f"\n[2/2] {prefix}: Running {len(missing_rl)} missing scenarios...")
                for i, scen in enumerate(missing_rl):
                    print(f"  [{i+1}/{len(missing_rl)}] {scen['network']:6s} | {scen['demand_type']:10s} | gw={scen['use_goodwill']} | bl={scen['backlog']}...", end="", flush=True)
                    res = _evaluate_rl_model(scen, rconfig, self.max_episodes, self.planning_horizon)
                    df_res = pd.DataFrame([res])
                    df_res.to_csv(rl_csv, mode='a', header=not os.path.exists(rl_csv), index=False)
                    
                    val = res[f'{prefix}_Profit']
                    if pd.isna(val):
                        print(f" DONE. Model missing? (NaN)")
                    else:
                        print(f" DONE. {prefix}_Profit={val:.1f}")

        # 3. Merge Output
        print("\n\n[3/3] Merging results into comprehensive dataset...")
        self.merge_outputs(all_scenarios)

    def merge_outputs(self, all_scenarios):
        baseline_csv = 'data/results/cache/benchmark_baselines_cache.csv'
        if not os.path.exists(baseline_csv):
            print("No baseline results found. Cannot merge.")
            return
            
        df_base = pd.read_csv(baseline_csv)
        merge_keys = ['Network', 'Demand', 'Goodwill', 'Backlog']
        
        # Merge RL caches
        for rconfig in RL_MODELS:
            prefix = rconfig['id']
            rl_csv = f'data/results/cache/benchmark_{prefix}_cache.csv'
            if os.path.exists(rl_csv):
                df_rl = pd.read_csv(rl_csv)
                df_base = pd.merge(df_base, df_rl, on=merge_keys, how='left')
        
        # Determine total scenario evaluation time
        time_cols = [c for c in df_base.columns if c.startswith('Time_Sec_')]
        df_base['Time_Sec'] = df_base[time_cols].sum(axis=1)

        output_file = 'data/results/benchmark_results_comprehensive_iterative.csv'
        df_base.to_csv(output_file, index=False)
        print(f"Iterative results saved to '{output_file}'.")
        
        # Quick CLI displays
        delta_cols = ['VPI', 'VSS', 'VPF_MSSP', 'VPF_DLP', 'VPF_Heuristic']
        df_ablation = df_base.groupby(['Network', 'Demand'])[delta_cols].mean()
        print("\nABLATION STUDY SUMMARY (Iterative)")
        print("=" * 80)
        print(df_ablation.to_string())

        print("\nRL COMPARISON — KEY SCENARIOS (Iterative)")
        print("=" * 80)
        focus_configs = [
            ('base',   'stationary', False, True),
            ('base',   'shock',      False, True),
            ('serial', 'trend',      False, True),
        ]
        
        available_rl_profs = [f"{rc['id']}_Profit" for rc in RL_MODELS if f"{rc['id']}_Profit" in df_base.columns]
        compare_cols = ['Network', 'Demand', 'Goodwill', 'Backlog',
                        'Oracle_Profit', 'MSSP_Profit', 'Heuristic_Profit'] + available_rl_profs
                        
        for net, dem, gw, bl in focus_configs:
            # We must carefully match datatypes since goodwill/backlog are booleans in CSV
            mask = ((df_base['Network'] == net) & (df_base['Demand'] == dem) & 
                    (df_base['Goodwill'] == gw) & (df_base['Backlog'] == bl))
            row = df_base.loc[mask, [c for c in compare_cols if c in df_base.columns]]
            if not row.empty:
                print(row.to_string(index=False))
                print("-" * 80)

if __name__ == "__main__":
    benchmark = BenchmarkSuite(max_episodes=5, planning_horizon=30)
    benchmark.run()
