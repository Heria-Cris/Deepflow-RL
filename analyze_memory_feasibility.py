# analyze_memory_feasibility.py
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from rl.envs.flow_env import DeepFlowEnv


# ============================================================
# Environment / PPO loading
# ============================================================

def build_base_env() -> DummyVecEnv:
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
    return model, env, model_path, stats_path


# ============================================================
# Helpers
# ============================================================

def set_scenario(raw_env: DeepFlowEnv, bw: float, lat: float, prompt_len: int):
    raw_env.reset(options={
        "bandwidth_mbps": bw,
        "latency_ms": lat,
        "prompt_len": prompt_len,
    })
    raw_env.set_scenario(bw, lat, prompt_len)


def predict_best_action(model: PPO, vec_env: VecNormalize, raw_env: DeepFlowEnv) -> List[int]:
    obs = raw_env._get_obs()
    norm_obs = vec_env.normalize_obs(obs.reshape(1, -1))
    action, _ = model.predict(norm_obs, deterministic=True)
    action = action[0] if len(action.shape) > 1 else action
    return [int(x) for x in action]


def action_to_real(raw_env: DeepFlowEnv, action: List[int]) -> Tuple[int, int, int]:
    mb_idx, k_idx, part = [int(x) for x in action]
    real_mb = raw_env.mb_options[mb_idx]
    real_k = raw_env.k_options[k_idx]
    return real_mb, real_k, part


def action_str(raw_env: DeepFlowEnv, action: List[int]) -> str:
    real_mb, real_k, part = action_to_real(raw_env, action)
    return f"P={part}, K={real_k}, MB={real_mb}"


def pct(x: int, total: int) -> float:
    return 100.0 * x / total if total > 0 else 0.0


# ============================================================
# Core analysis
# ============================================================

def enumerate_action_space(raw_env: DeepFlowEnv) -> List[Dict]:
    """
    枚举当前场景下所有动作，并返回分析记录。
    """
    records = []
    total_actions = raw_env.num_discrete_actions

    for flat_idx in range(total_actions):
        action = raw_env.unflatten_action(flat_idx)
        info = raw_env.evaluate_action(action)

        real_mb, real_k, real_p = action_to_real(raw_env, action)

        record = {
            "flat_idx": flat_idx,
            "action": action,
            "action_str": action_str(raw_env, action),
            "mb": real_mb,
            "k": real_k,
            "p": real_p,
            "valid": bool(info["valid"]),
            "feasible": bool(info["feasible"]),
            "throughput": float(info["throughput"]),
            "makespan": float(info["makespan"]),
            "edge_peak_memory_mb": float(info["edge_peak_memory_mb"]),
            "cloud_peak_memory_mb": float(info["cloud_peak_memory_mb"]),
            "edge_budget_mb": float(info["edge_budget_mb"]),
            "cloud_budget_mb": float(info["cloud_budget_mb"]),
            "oom_device": str(info["oom_device"]),
            "data_size_mb": float(info["data_size_mb"]),
        }
        records.append(record)

    return records


def summarize_feasibility(records: List[Dict]) -> Dict:
    total = len(records)
    valid = sum(r["valid"] for r in records)
    feasible = sum(r["valid"] and r["feasible"] for r in records)
    infeasible = sum(r["valid"] and (not r["feasible"]) for r in records)

    oom_counter = Counter()
    for r in records:
        if r["valid"] and (not r["feasible"]):
            oom_counter[r["oom_device"]] += 1

    return {
        "total": total,
        "valid": valid,
        "feasible": feasible,
        "infeasible": infeasible,
        "oom_counter": oom_counter,
    }


def print_feasibility_summary(title: str, records: List[Dict]):
    stats = summarize_feasibility(records)
    total = stats["total"]
    valid = stats["valid"]
    feasible = stats["feasible"]
    infeasible = stats["infeasible"]
    oom_counter = stats["oom_counter"]

    print(f"\n[{title}]")
    print(f"  - Total actions      : {total}")
    print(f"  - Valid actions      : {valid} ({pct(valid, total):.2f}%)")
    print(f"  - Feasible actions   : {feasible} ({pct(feasible, total):.2f}%)")
    print(f"  - Infeasible actions : {infeasible} ({pct(infeasible, total):.2f}%)")

    if infeasible > 0:
        print("  - OOM distribution:")
        for key in ["edge", "cloud", "both", "invalid_micro_batch", "none"]:
            if oom_counter.get(key, 0) > 0:
                print(f"      {key:<18}: {oom_counter[key]} ({pct(oom_counter[key], infeasible):.2f}% of infeasible)")
    else:
        print("  - OOM distribution   : no infeasible actions")


def top_feasible_actions(records: List[Dict], top_k: int = 10) -> List[Dict]:
    feasible_records = [r for r in records if r["valid"] and r["feasible"]]
    feasible_records.sort(key=lambda x: x["throughput"], reverse=True)
    return feasible_records[:top_k]


def print_top_feasible_actions(title: str, records: List[Dict], top_k: int = 10):
    top_records = top_feasible_actions(records, top_k=top_k)

    print(f"\n[{title}] Top-{top_k} feasible actions by throughput")
    print(f"{'Rank':<6} | {'Action':<18} | {'Throughput':<12} | {'Edge Mem':<12} | {'Cloud Mem':<12} | {'Data(MB)':<10}")
    print("-" * 88)

    for idx, r in enumerate(top_records, 1):
        print(
            f"{idx:<6} | "
            f"{r['action_str']:<18} | "
            f"{r['throughput']:<12.2f} | "
            f"{r['edge_peak_memory_mb']:<12.2f} | "
            f"{r['cloud_peak_memory_mb']:<12.2f} | "
            f"{r['data_size_mb']:<10.4f}"
        )


def print_top_infeasible_actions(title: str, records: List[Dict], top_k: int = 10):
    infeasible_records = [r for r in records if r["valid"] and (not r["feasible"])]
    infeasible_records.sort(
        key=lambda x: max(
            x["edge_peak_memory_mb"] / max(x["edge_budget_mb"], 1e-6),
            x["cloud_peak_memory_mb"] / max(x["cloud_budget_mb"], 1e-6),
        ),
        reverse=True,
    )

    print(f"\n[{title}] Top-{top_k} most memory-violating actions")
    print(f"{'Rank':<6} | {'Action':<18} | {'OOM Device':<12} | {'Edge Ratio':<12} | {'Cloud Ratio':<12}")
    print("-" * 76)

    for idx, r in enumerate(infeasible_records[:top_k], 1):
        edge_ratio = r["edge_peak_memory_mb"] / max(r["edge_budget_mb"], 1e-6)
        cloud_ratio = r["cloud_peak_memory_mb"] / max(r["cloud_budget_mb"], 1e-6)

        print(
            f"{idx:<6} | "
            f"{r['action_str']:<18} | "
            f"{r['oom_device']:<12} | "
            f"{edge_ratio:<12.3f} | "
            f"{cloud_ratio:<12.3f}"
        )

    if not infeasible_records:
        print("(no infeasible actions found)")


# ============================================================
# Requested analyses
# ============================================================

def experiment_1_single_scenario_feasibility(raw_env: DeepFlowEnv):
    """
    固定场景，枚举动作空间，统计 feasible / OOM 比例，打印 OOM 分布。
    """
    bw = 1.0
    lat = 50.0
    prompt_len = 2048

    set_scenario(raw_env, bw=bw, lat=lat, prompt_len=prompt_len)
    records = enumerate_action_space(raw_env)

    print("=" * 92)
    print("Experiment 1 - Enumerate action space in a fixed scenario")
    print("=" * 92)
    print(f"Scenario: bandwidth={bw} Mbps, latency={lat} ms, prompt={prompt_len}")

    print_feasibility_summary("Fixed Scenario Feasibility Summary", records)
    print_top_feasible_actions("Fixed Scenario", records, top_k=10)
    print_top_infeasible_actions("Fixed Scenario", records, top_k=10)


def experiment_2_compare_prompt_feasible_region(raw_env: DeepFlowEnv):
    """
    比较不同 prompt 下的可行域大小。
    """
    bw = 1.0
    lat = 50.0
    prompt_list = [128, 512, 1024, 2048, 4096]

    print("\n" + "=" * 92)
    print("Experiment 2 - Feasible region size under different prompt lengths")
    print("=" * 92)
    print(f"{'Prompt':<10} | {'Feasible':<10} | {'Infeasible':<12} | {'Feasible Ratio':<16} | {'Edge OOM':<10} | {'Cloud OOM':<10} | {'Both':<8}")
    print("-" * 100)

    for prompt_len in prompt_list:
        set_scenario(raw_env, bw=bw, lat=lat, prompt_len=prompt_len)
        records = enumerate_action_space(raw_env)
        stats = summarize_feasibility(records)

        oom_counter = stats["oom_counter"]
        print(
            f"{prompt_len:<10} | "
            f"{stats['feasible']:<10} | "
            f"{stats['infeasible']:<12} | "
            f"{pct(stats['feasible'], stats['total']):<16.2f}% | "
            f"{oom_counter.get('edge', 0):<10} | "
            f"{oom_counter.get('cloud', 0):<10} | "
            f"{oom_counter.get('both', 0):<8}"
        )


def experiment_3_check_current_ppo_memory_ratio(model: PPO, vec_env: VecNormalize, raw_env: DeepFlowEnv):
    """
    检查当前 PPO 动作的显存占用比例。
    """
    scenarios = [
        ("Weak / 512", 1.0, 50.0, 512),
        ("Weak / 2048", 1.0, 50.0, 2048),
        ("Weak / 4096", 1.0, 50.0, 4096),
        ("Strong / 512", 100.0, 10.0, 512),
        ("Strong / 2048", 100.0, 10.0, 2048),
        ("Strong / 4096", 100.0, 10.0, 4096),
    ]

    print("\n" + "=" * 92)
    print("Experiment 3 - Current PPO action memory ratio")
    print("=" * 92)
    print(f"{'Scenario':<16} | {'PPO Action':<18} | {'Feasible':<10} | {'Edge Ratio':<12} | {'Cloud Ratio':<12} | {'Throughput':<12}")
    print("-" * 96)

    for name, bw, lat, prompt_len in scenarios:
        set_scenario(raw_env, bw=bw, lat=lat, prompt_len=prompt_len)

        action = predict_best_action(model, vec_env, raw_env)
        info = raw_env.evaluate_action(action)

        edge_ratio = float(info["edge_peak_memory_mb"]) / max(float(info["edge_budget_mb"]), 1e-6)
        cloud_ratio = float(info["cloud_peak_memory_mb"]) / max(float(info["cloud_budget_mb"]), 1e-6)

        print(
            f"{name:<16} | "
            f"{action_str(raw_env, action):<18} | "
            f"{str(info['feasible']):<10} | "
            f"{edge_ratio:<12.3f} | "
            f"{cloud_ratio:<12.3f} | "
            f"{float(info['throughput']):<12.2f}"
        )


def experiment_4_promptwise_best_feasible_action(raw_env: DeepFlowEnv):
    """
    每个 prompt 下找最优 feasible 动作，看 memory 约束是否改变最优动作。
    """
    bw = 1.0
    lat = 50.0
    prompt_list = [128, 512, 1024, 2048, 4096]

    print("\n" + "=" * 92)
    print("Experiment 4 - Best feasible action under different prompt lengths")
    print("=" * 92)
    print(f"{'Prompt':<10} | {'Best Feasible Action':<20} | {'Throughput':<12} | {'Edge Ratio':<12} | {'Cloud Ratio':<12}")
    print("-" * 90)

    for prompt_len in prompt_list:
        set_scenario(raw_env, bw=bw, lat=lat, prompt_len=prompt_len)
        records = enumerate_action_space(raw_env)
        top_records = top_feasible_actions(records, top_k=1)

        if not top_records:
            print(f"{prompt_len:<10} | {'N/A':<20} | {'0.00':<12} | {'N/A':<12} | {'N/A':<12}")
            continue

        best = top_records[0]
        edge_ratio = best["edge_peak_memory_mb"] / max(best["edge_budget_mb"], 1e-6)
        cloud_ratio = best["cloud_peak_memory_mb"] / max(best["cloud_budget_mb"], 1e-6)

        print(
            f"{prompt_len:<10} | "
            f"{best['action_str']:<20} | "
            f"{best['throughput']:<12.2f} | "
            f"{edge_ratio:<12.3f} | "
            f"{cloud_ratio:<12.3f}"
        )


def experiment_5_oom_distribution_grid(raw_env: DeepFlowEnv):
    """
    统计不同 prompt 下 edge/cloud/both OOM 的数量分布。
    """
    bw = 1.0
    lat = 50.0
    prompt_list = [128, 512, 1024, 2048, 4096]

    print("\n" + "=" * 92)
    print("Experiment 5 - OOM distribution across prompt lengths")
    print("=" * 92)
    print(f"{'Prompt':<10} | {'Edge OOM':<10} | {'Cloud OOM':<10} | {'Both OOM':<10} | {'Feasible':<10} | {'Total':<10}")
    print("-" * 72)

    for prompt_len in prompt_list:
        set_scenario(raw_env, bw=bw, lat=lat, prompt_len=prompt_len)
        records = enumerate_action_space(raw_env)
        stats = summarize_feasibility(records)
        oom_counter = stats["oom_counter"]

        print(
            f"{prompt_len:<10} | "
            f"{oom_counter.get('edge', 0):<10} | "
            f"{oom_counter.get('cloud', 0):<10} | "
            f"{oom_counter.get('both', 0):<10} | "
            f"{stats['feasible']:<10} | "
            f"{stats['total']:<10}"
        )


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 92)
    print("DeepFlow-RL - Memory Feasibility / OOM Analysis")
    print("=" * 92)

    model, vec_env, model_path, stats_path = load_best_agent()
    raw_env = vec_env.envs[0]

    print(f"Loaded PPO model        : {model_path}")
    print(f"Loaded VecNormalize     : {stats_path}")
    print(f"Action space size       : {raw_env.num_discrete_actions}")
    print(f"MB options              : {raw_env.mb_options}")
    print(f"K options               : {raw_env.k_options}")
    print(f"Partition range         : [0, {raw_env.num_layers}]")
    print(f"Edge memory budget (MB) : {raw_env.edge.max_memory_gb * 1024 * raw_env.memory_budget_ratio:.2f}")
    print(f"Cloud memory budget (MB): {raw_env.cloud.max_memory_gb * 1024 * raw_env.memory_budget_ratio:.2f}")

    experiment_1_single_scenario_feasibility(raw_env)
    experiment_2_compare_prompt_feasible_region(raw_env)
    experiment_3_check_current_ppo_memory_ratio(model, vec_env, raw_env)
    experiment_4_promptwise_best_feasible_action(raw_env)
    experiment_5_oom_distribution_grid(raw_env)

    print("\n✅ Memory feasibility analysis completed.")


if __name__ == "__main__":
    main()