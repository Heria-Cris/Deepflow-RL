"""Run reproducible paper experiments with an explicitly paired PPO artifact."""

import argparse
import os
from pickle import UnpicklingError
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from policy_baselines import (
    INDEPENDENT_TEST_SCENARIOS,
    GlobalStaticSelection,
    heuristic_deepflow_action,
    search_per_scenario_oracle,
    select_global_static_deepflow,
    token_speculation_without_pipeline_action,
)
from result_tracking import ResultRecorder, build_run_metadata
from rl.envs.flow_env import DeepFlowEnv


# ============================================================
# Environment / Model Loading
# ============================================================

def build_base_env(config_dir: str = "configs"):
    env = DummyVecEnv([
        lambda: DeepFlowEnv(
            config_dir=config_dir,
            total_batch_size=32,
            episode_len=1,
            domain_randomization=False,
            seed=123,
            reward_mode="shaped",
        )
    ])
    env = VecMonitor(env)
    return env


def resolve_agent_artifacts(
    model_path: Optional[str] = None,
    vec_normalize_path: Optional[str] = None,
) -> Tuple[str, str]:
    """Select a model/statistics pair without permitting cross-run fallbacks."""
    if (model_path is None) != (vec_normalize_path is None):
        raise ValueError(
            "--model-path and --vec-normalize-path must be supplied together so that "
            "the PPO policy is evaluated with its own VecNormalize statistics."
        )

    if model_path is not None and vec_normalize_path is not None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"PPO model artifact not found: {model_path}")
        if not os.path.isfile(vec_normalize_path):
            raise FileNotFoundError(f"VecNormalize artifact not found: {vec_normalize_path}")
        return model_path, vec_normalize_path

    best_model_path = "models/ppo_deepflow/best_model/best_model.zip"
    best_stats_path = "models/ppo_deepflow/best_model/vec_normalize.pkl"
    final_model_path = "models/ppo_deepflow/final_model.zip"
    final_stats_path = "models/ppo_deepflow/vec_normalize.pkl"

    if os.path.exists(best_model_path) and os.path.exists(best_stats_path):
        return best_model_path, best_stats_path
    if os.path.exists(final_model_path) and os.path.exists(final_stats_path):
        return final_model_path, final_stats_path
    raise FileNotFoundError(
        "No default PPO/VecNormalize pair is available. Supply --model-path and "
        "--vec-normalize-path for a seed-specific evaluation."
    )


def load_best_agent(
    model_path: Optional[str] = None,
    vec_normalize_path: Optional[str] = None,
    config_dir: str = "configs",
):
    model_path, stats_path = resolve_agent_artifacts(model_path, vec_normalize_path)

    base_env = build_base_env(config_dir=config_dir)
    try:
        env = VecNormalize.load(stats_path, base_env)
    except (UnpicklingError, ValueError) as exc:
        raise RuntimeError(
            "The saved VecNormalize artifact cannot be used with the current environment. Ensure "
            "Git LFS has restored the binary file, then retrain PPO with train_phase5.py because "
            "the observation is now [bandwidth_mbps, link_delay_ms, prompt_len]. Use the paired "
            "model and VecNormalize artifact."
        ) from exc
    env.training = False
    env.norm_reward = False

    try:
        model = PPO.load(model_path, env=env)
    except ValueError as exc:
        raise RuntimeError(
            "The saved PPO model is incompatible with the current observation space. Retrain it "
            "with train_phase5.py and keep it paired with the matching VecNormalize artifact."
        ) from exc
    return model, env, model_path, stats_path


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


def _record_result(
    recorder: ResultRecorder,
    scenario_id: str,
    policy_name: str,
    raw_env: DeepFlowEnv,
    bandwidth_mbps: float,
    link_delay_ms: float,
    prompt_len: int,
    action: List[int],
    info: Dict,
):
    recorder.record(
        scenario_id=scenario_id,
        bandwidth_mbps=bandwidth_mbps,
        link_delay_ms=link_delay_ms,
        prompt_len=prompt_len,
        policy_name=policy_name,
        action=action,
        info=info,
        environment=raw_env,
    )


# ============================================================
# Feasible baseline search
# ============================================================

def _search_best_feasible_strict_local_target(raw_env: DeepFlowEnv) -> Tuple[Optional[List[int]], Optional[Dict]]:
    """
    Strict Local Target:
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


def _search_best_feasible_remote_target_without_speculation(raw_env: DeepFlowEnv) -> Tuple[Optional[List[int]], Optional[Dict]]:
    """
    Remote Target without Speculation:
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


# ============================================================
# Experiments
# ============================================================

def experiment_1_best_ppo_vs_feasible_baselines(
    model: PPO,
    vec_env: VecNormalize,
    raw_env: DeepFlowEnv,
    recorder: ResultRecorder,
    global_static: GlobalStaticSelection,
):
    print("\n[Experiment 1] Best PPO vs Best Feasible Baselines under Weak Network (1 Mbps, 50 ms, prompt=512)")
    print(f"{'Method':<26} | {'Strategy':<24} | {'Throughput(tok/s)':<20} | {'Speedup':<10}")
    print("-" * 100)

    _set_scenario(raw_env, bw=1.0, lat=50.0, prompt_len=512)

    local_action, local_info = _search_best_feasible_strict_local_target(raw_env)
    remote_action, remote_info = _search_best_feasible_remote_target_without_speculation(raw_env)
    split_action, split_info = _search_best_feasible_legacy_split(raw_env)
    no_pipeline_action = token_speculation_without_pipeline_action(raw_env)
    no_pipeline_info = _eval_action(raw_env, no_pipeline_action)
    global_static_action = list(global_static.action)
    global_static_info = _eval_action(raw_env, global_static_action)
    heuristic_action = heuristic_deepflow_action(raw_env, link_delay_ms=50.0, prompt_len=512)
    heuristic_info = _eval_action(raw_env, heuristic_action)
    ppo_action, ppo_info = _eval_best_action(model, vec_env, raw_env)
    oracle_action, oracle_info = search_per_scenario_oracle(raw_env)

    scenario_id = "experiment_1_1mbps_50ms_512"
    for policy_name, action, info in [
        ("strict_local_target", local_action, local_info),
        ("remote_target_without_speculation", remote_action, remote_info),
        ("best_feasible_legacy_activation_split", split_action, split_info),
        ("token_speculation_without_pipeline", no_pipeline_action, no_pipeline_info),
        ("global_static_deepflow", global_static_action, global_static_info),
        ("heuristic_deepflow", heuristic_action, heuristic_info),
        ("ppo_deepflow", ppo_action, ppo_info),
        ("per_scenario_oracle", oracle_action, oracle_info),
    ]:
        if action is not None and info is not None:
            _record_result(recorder, scenario_id, policy_name, raw_env, 1.0, 50.0, 512, action, info)

    # 用 best feasible local-only 作为 speedup 参考
    baseline_t = local_info["throughput"] if local_info is not None else 0.0

    if local_action is not None:
        _print_method_row("Best Feasible Strict Local Target", _action_str(raw_env, local_action), local_info["throughput"], baseline_t)
    else:
        _print_method_row("Best Feasible Strict Local Target", "N/A", 0.0, baseline_t)

    if remote_action is not None:
        _print_method_row("Best Feasible Remote Target without Speculation", _action_str(raw_env, remote_action), remote_info["throughput"], baseline_t)

    if split_action is not None:
        _print_method_row("Best Feasible Legacy Split", _action_str(raw_env, split_action), split_info["throughput"], baseline_t)

    _print_method_row("Token Speculation without Pipeline", _action_str(raw_env, no_pipeline_action), no_pipeline_info["throughput"], baseline_t)
    _print_method_row("Global Static DeepFlow", _action_str(raw_env, global_static_action), global_static_info["throughput"], baseline_t)
    _print_method_row("Heuristic DeepFlow", _action_str(raw_env, heuristic_action), heuristic_info["throughput"], baseline_t)

    _print_method_row("Best PPO (Ours)", _action_str(raw_env, ppo_action), ppo_info["throughput"], baseline_t)
    if oracle_action is not None:
        _print_method_row("Per-scenario Oracle", _action_str(raw_env, oracle_action), oracle_info["throughput"], baseline_t)

    return {
        "local_action": local_action,
        "local_info": local_info,
        "remote_action": remote_action,
        "remote_info": remote_info,
        "split_action": split_action,
        "split_info": split_info,
        "global_static_action": global_static_action,
        "global_static_info": global_static_info,
        "heuristic_action": heuristic_action,
        "heuristic_info": heuristic_info,
        "oracle_action": oracle_action,
        "oracle_info": oracle_info,
        "ppo_action": ppo_action,
        "ppo_info": ppo_info,
    }


def experiment_2_best_ppo_vs_best_static_deepflow(
    model: PPO,
    vec_env: VecNormalize,
    raw_env: DeepFlowEnv,
    recorder: ResultRecorder,
    global_static: GlobalStaticSelection,
):
    print("\n[Experiment 2] Global Static, Heuristic, PPO, and Oracle across representative scenarios")
    print(f"{'Scenario':<28} | {'Global Static':<14} | {'Heuristic':<12} | {'PPO':<12} | {'Oracle':<12}")
    print("-" * 102)

    scenarios = [
        ("Weak Net / Short Prompt", 1.0, 50.0, 512),
        ("Moderate Net / Short Prompt", 5.0, 30.0, 512),
        ("Strong Net / Short Prompt", 100.0, 10.0, 512),
        ("Weak Net / Long Prompt", 1.0, 50.0, 1536),
        ("Strong Net / Long Prompt", 100.0, 10.0, 1536),
    ]

    for scenario_index, (name, bw, lat, prompt_len) in enumerate(scenarios, 1):
        _set_scenario(raw_env, bw=bw, lat=lat, prompt_len=prompt_len)

        global_static_action = list(global_static.action)
        global_static_info = _eval_action(raw_env, global_static_action)
        heuristic_action = heuristic_deepflow_action(raw_env, link_delay_ms=lat, prompt_len=prompt_len)
        heuristic_info = _eval_action(raw_env, heuristic_action)
        best_action, info_best = _eval_best_action(model, vec_env, raw_env)
        oracle_action, oracle_info = search_per_scenario_oracle(raw_env)

        scenario_id = f"experiment_2_{scenario_index}"
        _record_result(recorder, scenario_id, "global_static_deepflow", raw_env, bw, lat, prompt_len, global_static_action, global_static_info)
        _record_result(recorder, scenario_id, "heuristic_deepflow", raw_env, bw, lat, prompt_len, heuristic_action, heuristic_info)
        _record_result(
            recorder, scenario_id, "ppo_deepflow",
            raw_env, bw, lat, prompt_len, best_action, info_best,
        )
        if oracle_action is not None and oracle_info is not None:
            _record_result(recorder, scenario_id, "per_scenario_oracle", raw_env, bw, lat, prompt_len, oracle_action, oracle_info)
        print(
            f"{name:<28} | "
            f"{global_static_info['throughput']:<14.2f} | "
            f"{heuristic_info['throughput']:<12.2f} | "
            f"{info_best['throughput']:<12.2f} | "
            f"{(oracle_info['throughput'] if oracle_info is not None else 0.0):<12.2f}"
        )


def experiment_3_bandwidth_sensitivity_best_ppo(
    model: PPO,
    vec_env: VecNormalize,
    raw_env: DeepFlowEnv,
    recorder: ResultRecorder,
    global_static: GlobalStaticSelection,
):
    print("\n[Experiment 3] Bandwidth Sensitivity: Legacy, Global Static, Heuristic, PPO, and Oracle")
    print(f"{'BW(Mbps)':<10} | {'Legacy':<12} | {'Global Static':<14} | {'Heuristic':<12} | {'PPO':<12} | {'Oracle':<12}")
    print("-" * 100)

    bandwidths = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]

    for scenario_index, bw in enumerate(bandwidths, 1):
        _set_scenario(raw_env, bw=bw, lat=50.0, prompt_len=512)

        split_action, split_info = _search_best_feasible_legacy_split(raw_env)
        global_static_action = list(global_static.action)
        global_static_info = _eval_action(raw_env, global_static_action)
        heuristic_action = heuristic_deepflow_action(raw_env, link_delay_ms=50.0, prompt_len=512)
        heuristic_info = _eval_action(raw_env, heuristic_action)
        best_action, info_best = _eval_best_action(model, vec_env, raw_env)
        oracle_action, oracle_info = search_per_scenario_oracle(raw_env)

        split_t = split_info["throughput"] if split_info is not None else 0.0
        scenario_id = f"experiment_3_bw_{scenario_index}"
        if split_action is not None and split_info is not None:
            _record_result(
                recorder, scenario_id, "best_feasible_legacy_activation_split",
                raw_env, bw, 50.0, 512, split_action, split_info,
            )
        _record_result(recorder, scenario_id, "global_static_deepflow", raw_env, bw, 50.0, 512, global_static_action, global_static_info)
        _record_result(recorder, scenario_id, "heuristic_deepflow", raw_env, bw, 50.0, 512, heuristic_action, heuristic_info)
        _record_result(
            recorder, scenario_id, "ppo_deepflow",
            raw_env, bw, 50.0, 512, best_action, info_best,
        )
        if oracle_action is not None and oracle_info is not None:
            _record_result(recorder, scenario_id, "per_scenario_oracle", raw_env, bw, 50.0, 512, oracle_action, oracle_info)

        print(
            f"{bw:<10} | "
            f"{split_t:<12.2f} | "
            f"{global_static_info['throughput']:<14.2f} | "
            f"{heuristic_info['throughput']:<12.2f} | "
            f"{info_best['throughput']:<12.2f} | "
            f"{(oracle_info['throughput'] if oracle_info is not None else 0.0):<12.2f}"
        )


def experiment_4_action_sensitivity_table(
    model: PPO,
    vec_env: VecNormalize,
    raw_env: DeepFlowEnv,
    recorder: ResultRecorder,
):
    print("\n[Experiment 4] Action Sensitivity Analysis (Does the PPO action change with scenario?)")
    print(
        f"{'Scenario':<28} | {'BW':<8} | {'Lat(ms)':<8} | {'Prompt':<8} | "
        f"{'Chosen Action':<22} | {'Throughput':<12} | {'Changed?':<10}"
    )
    print("-" * 125)

    scenarios = [
        ("Weak / 128", 1.0, 50.0, 128),
        ("Weak / 512", 1.0, 50.0, 512),
        ("Weak / 1024", 1.0, 50.0, 1024),
        ("Moderate / 512", 5.0, 30.0, 512),
        ("Strong / 512", 100.0, 10.0, 512),
        ("Strong / 1536", 100.0, 10.0, 1536),
        ("Very Weak / 512", 0.5, 80.0, 512),
        ("High RTT / 512", 10.0, 120.0, 512),
    ]

    prev_action = None
    unique_actions = set()

    for scenario_index, (name, bw, lat, prompt_len) in enumerate(scenarios, 1):
        _set_scenario(raw_env, bw=bw, lat=lat, prompt_len=prompt_len)
        action, info = _eval_best_action(model, vec_env, raw_env)

        action_real = _action_str(raw_env, action)
        unique_actions.add(action_real)

        changed = "N/A" if prev_action is None else ("Yes" if action != prev_action else "No")
        prev_action = action

        _record_result(
            recorder, f"experiment_4_{scenario_index}", "ppo_deepflow",
            raw_env, bw, lat, prompt_len, action, info,
        )

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


def experiment_5_action_distribution_grid(
    model: PPO,
    vec_env: VecNormalize,
    raw_env: DeepFlowEnv,
    recorder: ResultRecorder,
):
    print("\n[Experiment 5] PPO Action Grid over Network/Prompt Conditions")
    print(f"{'BW':<8} | {'Lat':<8} | {'Prompt':<8} | {'PPO Action':<22} | {'Throughput':<12}")
    print("-" * 78)

    bandwidths = [0.5, 1.0, 10.0, 100.0]
    latencies = [10.0, 50.0, 100.0]
    prompts = [128, 512, 1024, 1536]

    scenario_index = 0
    for bw in bandwidths:
        for lat in latencies:
            for prompt_len in prompts:
                scenario_index += 1
                _set_scenario(raw_env, bw=bw, lat=lat, prompt_len=prompt_len)
                action, info = _eval_best_action(model, vec_env, raw_env)

                _record_result(
                    recorder, f"experiment_5_{scenario_index}", "ppo_deepflow",
                    raw_env, bw, lat, prompt_len, action, info,
                )

                print(
                    f"{bw:<8.1f} | "
                    f"{lat:<8.1f} | "
                    f"{prompt_len:<8d} | "
                    f"{_action_str(raw_env, action):<22} | "
                    f"{info['throughput']:<12.2f}"
                )


def experiment_6_independent_policy_comparison(
    model: PPO,
    vec_env: VecNormalize,
    raw_env: DeepFlowEnv,
    recorder: ResultRecorder,
    global_static: GlobalStaticSelection,
):
    print("\n[Experiment 6] Independent Test: Fixed, Heuristic, PPO, and Per-scenario Oracle")
    print(f"{'Scenario':<30} | {'Global Static':<14} | {'Heuristic':<12} | {'PPO':<12} | {'Oracle':<12}")
    print("-" * 106)

    for scenario_index, (name, bw, lat, prompt_len) in enumerate(INDEPENDENT_TEST_SCENARIOS, 1):
        _set_scenario(raw_env, bw=bw, lat=lat, prompt_len=prompt_len)

        local_action, local_info = _search_best_feasible_strict_local_target(raw_env)
        remote_action, remote_info = _search_best_feasible_remote_target_without_speculation(raw_env)
        legacy_action, legacy_info = _search_best_feasible_legacy_split(raw_env)
        no_pipeline_action = token_speculation_without_pipeline_action(raw_env)
        no_pipeline_info = _eval_action(raw_env, no_pipeline_action)
        global_static_action = list(global_static.action)
        global_static_info = _eval_action(raw_env, global_static_action)
        heuristic_action = heuristic_deepflow_action(raw_env, link_delay_ms=lat, prompt_len=prompt_len)
        heuristic_info = _eval_action(raw_env, heuristic_action)
        ppo_action, ppo_info = _eval_best_action(model, vec_env, raw_env)
        oracle_action, oracle_info = search_per_scenario_oracle(raw_env)

        scenario_id = f"independent_test_{scenario_index}"
        for policy_name, action, info in [
            ("strict_local_target", local_action, local_info),
            ("remote_target_without_speculation", remote_action, remote_info),
            ("best_feasible_legacy_activation_split", legacy_action, legacy_info),
            ("token_speculation_without_pipeline", no_pipeline_action, no_pipeline_info),
            ("global_static_deepflow", global_static_action, global_static_info),
            ("heuristic_deepflow", heuristic_action, heuristic_info),
            ("ppo_deepflow", ppo_action, ppo_info),
            ("per_scenario_oracle", oracle_action, oracle_info),
        ]:
            if action is not None and info is not None:
                _record_result(recorder, scenario_id, policy_name, raw_env, bw, lat, prompt_len, action, info)

        print(
            f"{name:<28} | "
            f"{global_static_info['throughput']:<14.2f} | "
            f"{heuristic_info['throughput']:<12.2f} | "
            f"{ppo_info['throughput']:<12.2f} | "
            f"{(oracle_info['throughput'] if oracle_info is not None else 0.0):<12.2f}"
        )


# ============================================================
# Main
# ============================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="Seed-specific PPO .zip artifact path.")
    parser.add_argument("--vec-normalize-path", help="Paired seed-specific VecNormalize .pkl path.")
    parser.add_argument("--output-dir", default=os.path.join("results", "paper_experiments"))
    parser.add_argument("--suite-name", default="paper_experiments")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument(
        "--only-independent-test",
        action="store_true",
        help="Run Global Static calibration and the 32-scenario independent test only.",
    )
    args = parser.parse_args(argv)
    if (args.model_path is None) != (args.vec_normalize_path is None):
        parser.error("--model-path and --vec-normalize-path must be specified together")
    return args


def run_paper_experiments(args: Optional[argparse.Namespace] = None):
    args = args or parse_args()
    print("=" * 86)
    print("📊 DeepFlow-RL - Unified Paper Experiments (Best Feasible Baselines + PPO)")
    print("=" * 86)

    model, vec_env, model_path, stats_path = load_best_agent(
        model_path=args.model_path,
        vec_normalize_path=args.vec_normalize_path,
        config_dir=args.config_dir,
    )
    raw_env = vec_env.envs[0]
    recorder = ResultRecorder(
        output_dir=args.output_dir,
        metadata=build_run_metadata(
            suite=args.suite_name,
            config_dir=raw_env.config_dir,
            total_batch_size=raw_env.total_batch_size,
            pressure_profile="base",
            random_seed=args.random_seed,
            ppo_model_path=model_path,
            vec_normalize_path=stats_path,
        ),
    )
    global_static = select_global_static_deepflow(raw_env)
    recorder.update_metadata(
        global_static_deepflow=global_static.as_metadata(),
        independent_test_scenarios=[
            {
                "scenario_id": scenario_id,
                "bandwidth_mbps": bandwidth,
                "link_delay_ms": link_delay,
                "prompt_len": prompt_len,
            }
            for scenario_id, bandwidth, link_delay, prompt_len in INDEPENDENT_TEST_SCENARIOS
        ],
    )
    print(
        "Global Static calibration: "
        f"{_action_str(raw_env, list(global_static.action))}, "
        f"mean Oracle ratio={global_static.mean_oracle_ratio:.4f}"
    )

    if not args.only_independent_test:
        experiment_1_best_ppo_vs_feasible_baselines(model, vec_env, raw_env, recorder, global_static)
        experiment_2_best_ppo_vs_best_static_deepflow(model, vec_env, raw_env, recorder, global_static)
        experiment_3_bandwidth_sensitivity_best_ppo(model, vec_env, raw_env, recorder, global_static)
        experiment_4_action_sensitivity_table(model, vec_env, raw_env, recorder)
        experiment_5_action_distribution_grid(model, vec_env, raw_env, recorder)
    experiment_6_independent_policy_comparison(model, vec_env, raw_env, recorder, global_static)

    json_path, csv_path = recorder.write()
    print(f"Structured JSON results: {json_path}")
    print(f"Structured CSV results : {csv_path}")

    print("\n✅ All experiments completed.")


if __name__ == "__main__":
    run_paper_experiments(parse_args())
