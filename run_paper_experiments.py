# run_paper_experiments.py
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from rl.envs.flow_env import DeepFlowEnv


# ============================================================
# Environment / Model Loading
# ============================================================

def build_base_env():
    env = DummyVecEnv([
        lambda: DeepFlowEnv(
            config_dir="configs",
            total_batch_size=32,
            episode_len=1,
            domain_randomization=False,
            seed=123,
            reward_mode="shaped",
        )
    ])
    env = VecMonitor(env)
    return env


def load_best_agent():
    best_model_path = "models/ppo_deepflow/best_model/best_model.zip"
    best_stats_path = "models/ppo_deepflow/best_model/vec_normalize.pkl"
    final_model_path = "models/ppo_deepflow/final_model.zip"
    final_stats_path = "models/ppo_deepflow/vec_normalize.pkl"

    if os.path.exists(best_model_path) and os.path.exists(best_stats_path):
        model_path = best_model_path
        stats_path = best_stats_path
    else:
        model_path = final_model_path
        stats_path = final_stats_path

    base_env = build_base_env()
    env = VecNormalize.load(stats_path, base_env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)
    return model, env


# ============================================================
# Helpers
# ============================================================

def _set_scenario(raw_env: DeepFlowEnv, bw: float, lat: float, prompt_len: int):
    raw_env.reset(options={
        "bandwidth_mbps": bw,
        "latency_ms": lat,
        "prompt_len": prompt_len,
    })
    raw_env.set_scenario(bw, lat, prompt_len)


def _predict_best_action(model: PPO, vec_env: VecNormalize, raw_env: DeepFlowEnv):
    obs = raw_env._get_obs()
    norm_obs = vec_env.normalize_obs(obs.reshape(1, -1))
    action, _ = model.predict(norm_obs, deterministic=True)
    action = action[0] if len(action.shape) > 1 else action
    return [int(x) for x in action]


def _eval_action(raw_env: DeepFlowEnv, action: List[int]) -> Dict:
    return raw_env.evaluate_action(action)


def _eval_best_action(model: PPO, vec_env: VecNormalize, raw_env: DeepFlowEnv) -> Tuple[List[int], Dict]:
    action = _predict_best_action(model, vec_env, raw_env)
    info = raw_env.evaluate_action(action)
    return action, info


def _action_to_real(raw_env: DeepFlowEnv, action: List[int]) -> Tuple[int, int, int]:
    mb_idx, k_idx, part = [int(x) for x in action]
    real_mb = raw_env.mb_options[mb_idx]
    real_k = raw_env.k_options[k_idx]
    return real_mb, real_k, part


def _action_str(raw_env: DeepFlowEnv, action: List[int]) -> str:
    real_mb, real_k, part = _action_to_real(raw_env, action)
    return f"P={part}, K={real_k}, MB={real_mb}"


def _format_speedup(x: float, base: float) -> str:
    if base <= 1e-12:
        return "N/A"
    return f"{x / base:.2f}x"


def _print_method_row(method: str, strategy: str, throughput: float, speedup_base: float):
    print(f"{method:<26} | {strategy:<24} | {throughput:<20.2f} | {_format_speedup(throughput, speedup_base):<10}")


# ============================================================
# Feasible baseline search
# ============================================================

def _search_best_feasible_local_only(raw_env: DeepFlowEnv) -> Tuple[Optional[List[int]], Optional[Dict]]:
    """
    Local only:
      - partition = all target layers on edge => P=num_layers
      - k = 0
      - search MB
    """
    best_action = None
    best_info = None
    best_t = -1.0

    for mb_idx in range(len(raw_env.mb_options)):
        action = [mb_idx, 0, raw_env.num_layers]   # K=0, P=32
        info = _eval_action(raw_env, action)
        if info["valid"] and info["feasible"] and info["throughput"] > best_t:
            best_t = info["throughput"]
            best_action = action
            best_info = info

    return best_action, best_info


def _search_best_feasible_cloud_only(raw_env: DeepFlowEnv) -> Tuple[Optional[List[int]], Optional[Dict]]:
    """
    Cloud only:
      - partition = 0
      - k = 0
      - search MB
    """
    best_action = None
    best_info = None
    best_t = -1.0

    for mb_idx in range(len(raw_env.mb_options)):
        action = [mb_idx, 0, 0]   # K=0, P=0
        info = _eval_action(raw_env, action)
        if info["valid"] and info["feasible"] and info["throughput"] > best_t:
            best_t = info["throughput"]
            best_action = action
            best_info = info

    return best_action, best_info


def _search_best_feasible_legacy_split(raw_env: DeepFlowEnv) -> Tuple[Optional[List[int]], Optional[Dict]]:
    """
    Legacy split:
      - k = 0
      - partition in [1, num_layers-1]
      - search MB and partition
    """
    best_action = None
    best_info = None
    best_t = -1.0

    for mb_idx in range(len(raw_env.mb_options)):
        for p in range(1, raw_env.num_layers):
            action = [mb_idx, 0, p]
            info = _eval_action(raw_env, action)
            if info["valid"] and info["feasible"] and info["throughput"] > best_t:
                best_t = info["throughput"]
                best_action = action
                best_info = info

    return best_action, best_info


def _search_best_static_deepflow(raw_env: DeepFlowEnv) -> Tuple[Optional[List[int]], Optional[Dict]]:
    """
    Static DeepFlow baseline:
      - partition fixed at 0
      - search MB and K statically
      - no RL
    """
    best_action = None
    best_info = None
    best_t = -1.0

    for mb_idx in range(len(raw_env.mb_options)):
        for k_idx in range(len(raw_env.k_options)):
            action = [mb_idx, k_idx, 0]
            info = _eval_action(raw_env, action)
            if info["valid"] and info["feasible"] and info["throughput"] > best_t:
                best_t = info["throughput"]
                best_action = action
                best_info = info

    return best_action, best_info


# ============================================================
# Experiments
# ============================================================

def experiment_1_best_ppo_vs_feasible_baselines(model: PPO, vec_env: VecNormalize, raw_env: DeepFlowEnv):
    print("\n[Experiment 1] Best PPO vs Best Feasible Baselines under Weak Network (1 Mbps, 50 ms, prompt=512)")
    print(f"{'Method':<26} | {'Strategy':<24} | {'Throughput(tok/s)':<20} | {'Speedup':<10}")
    print("-" * 100)

    _set_scenario(raw_env, bw=1.0, lat=50.0, prompt_len=512)

    local_action, local_info = _search_best_feasible_local_only(raw_env)
    cloud_action, cloud_info = _search_best_feasible_cloud_only(raw_env)
    split_action, split_info = _search_best_feasible_legacy_split(raw_env)
    static_action, static_info = _search_best_static_deepflow(raw_env)
    ppo_action, ppo_info = _eval_best_action(model, vec_env, raw_env)

    # 用 best feasible local-only 作为 speedup 参考
    baseline_t = local_info["throughput"] if local_info is not None else 0.0

    if local_action is not None:
        _print_method_row("Best Feasible Local Only", _action_str(raw_env, local_action), local_info["throughput"], baseline_t)
    else:
        _print_method_row("Best Feasible Local Only", "N/A", 0.0, baseline_t)

    if cloud_action is not None:
        _print_method_row("Best Feasible Cloud Only", _action_str(raw_env, cloud_action), cloud_info["throughput"], baseline_t)

    if split_action is not None:
        _print_method_row("Best Feasible Legacy Split", _action_str(raw_env, split_action), split_info["throughput"], baseline_t)

    if static_action is not None:
        _print_method_row("Best Static DeepFlow", _action_str(raw_env, static_action), static_info["throughput"], baseline_t)

    _print_method_row("Best PPO (Ours)", _action_str(raw_env, ppo_action), ppo_info["throughput"], baseline_t)

    return {
        "local_action": local_action,
        "local_info": local_info,
        "cloud_action": cloud_action,
        "cloud_info": cloud_info,
        "split_action": split_action,
        "split_info": split_info,
        "static_action": static_action,
        "static_info": static_info,
        "ppo_action": ppo_action,
        "ppo_info": ppo_info,
    }


def experiment_2_best_ppo_vs_best_static_deepflow(model: PPO, vec_env: VecNormalize, raw_env: DeepFlowEnv):
    print("\n[Experiment 2] Best PPO vs Best Static DeepFlow across representative scenarios")
    print(f"{'Scenario':<28} | {'Best Static DeepFlow':<20} | {'Best PPO':<12} | {'PPO Action':<22}")
    print("-" * 100)

    scenarios = [
        ("Weak Net / Short Prompt", 1.0, 50.0, 512),
        ("Moderate Net / Short Prompt", 5.0, 30.0, 512),
        ("Strong Net / Short Prompt", 100.0, 10.0, 512),
        ("Weak Net / Long Prompt", 1.0, 50.0, 2048),
        ("Strong Net / Long Prompt", 100.0, 10.0, 2048),
    ]

    for name, bw, lat, prompt_len in scenarios:
        _set_scenario(raw_env, bw=bw, lat=lat, prompt_len=prompt_len)

        static_action, static_info = _search_best_static_deepflow(raw_env)
        best_action, info_best = _eval_best_action(model, vec_env, raw_env)

        static_t = static_info["throughput"] if static_info is not None else 0.0
        print(
            f"{name:<28} | "
            f"{static_t:<20.2f} | "
            f"{info_best['throughput']:<12.2f} | "
            f"{_action_str(raw_env, best_action):<22}"
        )


def experiment_3_bandwidth_sensitivity_best_ppo(model: PPO, vec_env: VecNormalize, raw_env: DeepFlowEnv):
    print("\n[Experiment 3] Bandwidth Sensitivity: Best PPO vs Best Feasible Baselines (prompt=512, latency=50 ms)")
    print(f"{'BW(Mbps)':<10} | {'Best Legacy Split':<18} | {'Best Static DeepFlow':<20} | {'Best PPO':<12} | {'PPO Action':<22}")
    print("-" * 115)

    bandwidths = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]

    for bw in bandwidths:
        _set_scenario(raw_env, bw=bw, lat=50.0, prompt_len=512)

        split_action, split_info = _search_best_feasible_legacy_split(raw_env)
        static_action, static_info = _search_best_static_deepflow(raw_env)
        best_action, info_best = _eval_best_action(model, vec_env, raw_env)

        split_t = split_info["throughput"] if split_info is not None else 0.0
        static_t = static_info["throughput"] if static_info is not None else 0.0

        print(
            f"{bw:<10} | "
            f"{split_t:<18.2f} | "
            f"{static_t:<20.2f} | "
            f"{info_best['throughput']:<12.2f} | "
            f"{_action_str(raw_env, best_action):<22}"
        )


def experiment_4_action_sensitivity_table(model: PPO, vec_env: VecNormalize, raw_env: DeepFlowEnv):
    print("\n[Experiment 4] Action Sensitivity Analysis (Does the PPO action change with scenario?)")
    print(
        f"{'Scenario':<28} | {'BW':<8} | {'Lat(ms)':<8} | {'Prompt':<8} | "
        f"{'Chosen Action':<22} | {'Throughput':<12} | {'Changed?':<10}"
    )
    print("-" * 125)

    scenarios = [
        ("Weak / 128", 1.0, 50.0, 128),
        ("Weak / 512", 1.0, 50.0, 512),
        ("Weak / 2048", 1.0, 50.0, 2048),
        ("Moderate / 512", 5.0, 30.0, 512),
        ("Strong / 512", 100.0, 10.0, 512),
        ("Strong / 2048", 100.0, 10.0, 2048),
        ("Very Weak / 512", 0.5, 80.0, 512),
        ("High RTT / 512", 10.0, 120.0, 512),
    ]

    prev_action = None
    unique_actions = set()

    for name, bw, lat, prompt_len in scenarios:
        _set_scenario(raw_env, bw=bw, lat=lat, prompt_len=prompt_len)
        action, info = _eval_best_action(model, vec_env, raw_env)

        action_real = _action_str(raw_env, action)
        unique_actions.add(action_real)

        changed = "N/A" if prev_action is None else ("Yes" if action != prev_action else "No")
        prev_action = action

        print(
            f"{name:<28} | "
            f"{bw:<8.1f} | "
            f"{lat:<8.1f} | "
            f"{prompt_len:<8d} | "
            f"{action_real:<22} | "
            f"{info['throughput']:<12.2f} | "
            f"{changed:<10}"
        )

    print("-" * 125)
    print(f"Unique PPO actions across scenarios: {len(unique_actions)}")
    for idx, action_name in enumerate(sorted(unique_actions), 1):
        print(f"  {idx}. {action_name}")


def experiment_5_action_distribution_grid(model: PPO, vec_env: VecNormalize, raw_env: DeepFlowEnv):
    print("\n[Experiment 5] PPO Action Grid over Network/Prompt Conditions")
    print(f"{'BW':<8} | {'Lat':<8} | {'Prompt':<8} | {'PPO Action':<22} | {'Throughput':<12}")
    print("-" * 78)

    bandwidths = [0.5, 1.0, 10.0, 100.0]
    latencies = [10.0, 50.0, 100.0]
    prompts = [128, 512, 2048]

    for bw in bandwidths:
        for lat in latencies:
            for prompt_len in prompts:
                _set_scenario(raw_env, bw=bw, lat=lat, prompt_len=prompt_len)
                action, info = _eval_best_action(model, vec_env, raw_env)

                print(
                    f"{bw:<8.1f} | "
                    f"{lat:<8.1f} | "
                    f"{prompt_len:<8d} | "
                    f"{_action_str(raw_env, action):<22} | "
                    f"{info['throughput']:<12.2f}"
                )


def experiment_6_compare_ppo_to_feasible_oracle(model: PPO, vec_env: VecNormalize, raw_env: DeepFlowEnv):
    """
    新增：比较 PPO 与当前场景下的 feasible oracle（全动作枚举最优）。
    """
    print("\n[Experiment 6] Best PPO vs Feasible Oracle")
    print(f"{'Scenario':<28} | {'Feasible Oracle':<18} | {'Best PPO':<12} | {'Oracle Action':<22} | {'PPO Action':<22}")
    print("-" * 125)

    scenarios = [
        ("Weak / 128", 1.0, 50.0, 128),
        ("Weak / 512", 1.0, 50.0, 512),
        ("Weak / 2048", 1.0, 50.0, 2048),
        ("Weak / 4096", 1.0, 50.0, 4096),
        ("Strong / 512", 100.0, 10.0, 512),
        ("Strong / 2048", 100.0, 10.0, 2048),
    ]

    for name, bw, lat, prompt_len in scenarios:
        _set_scenario(raw_env, bw=bw, lat=lat, prompt_len=prompt_len)

        # brute-force feasible oracle
        best_oracle_action = None
        best_oracle_info = None
        best_t = -1.0
        for idx in range(raw_env.num_discrete_actions):
            action = raw_env.unflatten_action(idx)
            info = _eval_action(raw_env, action)
            if info["valid"] and info["feasible"] and info["throughput"] > best_t:
                best_t = info["throughput"]
                best_oracle_action = action
                best_oracle_info = info

        ppo_action, ppo_info = _eval_best_action(model, vec_env, raw_env)

        oracle_t = best_oracle_info["throughput"] if best_oracle_info is not None else 0.0

        print(
            f"{name:<28} | "
            f"{oracle_t:<18.2f} | "
            f"{ppo_info['throughput']:<12.2f} | "
            f"{_action_str(raw_env, best_oracle_action):<22} | "
            f"{_action_str(raw_env, ppo_action):<22}"
        )


# ============================================================
# Main
# ============================================================

def run_paper_experiments():
    print("=" * 86)
    print("📊 DeepFlow-RL - Unified Paper Experiments (Best Feasible Baselines + PPO)")
    print("=" * 86)

    model, vec_env = load_best_agent()
    raw_env = vec_env.envs[0]

    experiment_1_best_ppo_vs_feasible_baselines(model, vec_env, raw_env)
    experiment_2_best_ppo_vs_best_static_deepflow(model, vec_env, raw_env)
    experiment_3_bandwidth_sensitivity_best_ppo(model, vec_env, raw_env)
    experiment_4_action_sensitivity_table(model, vec_env, raw_env)
    experiment_5_action_distribution_grid(model, vec_env, raw_env)
    experiment_6_compare_ppo_to_feasible_oracle(model, vec_env, raw_env)

    print("\n✅ All experiments completed.")


if __name__ == "__main__":
    run_paper_experiments()