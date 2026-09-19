"""Run common-random-number weak-network and resource stress experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from engine.stress import STRESS_PROFILES, StressProfile, generate_network_trials, get_stress_profile
from policy_baselines import (
    GlobalStaticSelection,
    heuristic_deepflow_action,
    select_global_static_deepflow,
    token_speculation_without_pipeline_action,
)
from result_tracking import ResultRecorder, build_run_metadata
from rl.envs.flow_env import DeepFlowEnv
from run_paper_experiments import _predict_best_action, load_best_agent


StressScenario = Tuple[str, float, float, int]
STRESS_SCENARIOS: Tuple[StressScenario, ...] = tuple(
    (
        f"stress_bw{bandwidth:g}_lat{link_delay:g}_p{prompt_len}",
        bandwidth,
        link_delay,
        prompt_len,
    )
    for bandwidth in (0.5, 1.0, 5.0, 10.0)
    for link_delay in (50.0, 80.0, 120.0)
    for prompt_len in (128, 512, 1024, 1536)
)


@dataclass(frozen=True)
class ProfileActionSelection:
    action: Tuple[int, int, int]
    mean_throughput_tok_s: float
    feasible: bool

    def as_metadata(self, environment: DeepFlowEnv) -> Dict[str, Any]:
        mb_idx, k_idx, partition_point = self.action
        return {
            "action": list(self.action),
            "micro_batch_size": environment.mb_options[mb_idx],
            "k_steps": environment.k_options[k_idx],
            "partition_point": partition_point,
            "mean_throughput_tok_s": self.mean_throughput_tok_s,
            "feasible": self.feasible,
        }


def _stable_scenario_seed(master_seed: int, profile_name: str, scenario_id: str) -> int:
    source = f"{master_seed}|{profile_name}|{scenario_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(source).digest()[:8], byteorder="big") % (2 ** 32)


def _set_scenario(environment: DeepFlowEnv, bandwidth_mbps: float, link_delay_ms: float, prompt_len: int) -> None:
    environment.reset(options={
        "bandwidth_mbps": bandwidth_mbps,
        "latency_ms": link_delay_ms,
        "prompt_len": prompt_len,
    })
    environment.set_scenario(bandwidth_mbps, link_delay_ms, prompt_len)


def _trial_infos(
    environment: DeepFlowEnv,
    action: Sequence[int],
    profile: StressProfile,
    network_trials,
) -> List[Dict[str, Any]]:
    micro_batch_size = environment.mb_options[int(action[0])]
    micro_batch_count = environment.total_batch_size // micro_batch_size
    if micro_batch_count <= 0:
        raise ValueError("Action has an invalid micro-batch size")
    return environment.evaluate_action_trials(
        action,
        stress_profile=profile,
        network_trials=[trial[:micro_batch_count] for trial in network_trials],
    )


def _mean_throughput(infos: Iterable[Mapping[str, Any]]) -> float:
    values = [float(info["throughput"]) for info in infos]
    return float(mean(values)) if values else 0.0


def select_profile_expected_action(
    environment: DeepFlowEnv,
    candidate_actions: Iterable[Sequence[int]],
    profile: StressProfile,
    network_trials,
) -> ProfileActionSelection:
    """Choose an action by common-trial expected throughput without future-loss lookahead."""
    best: Optional[ProfileActionSelection] = None
    best_flat_index: Optional[int] = None

    for candidate in candidate_actions:
        action = tuple(int(value) for value in candidate)
        infos = _trial_infos(environment, action, profile, network_trials)
        feasible = all(bool(info["valid"]) and bool(info["feasible"]) for info in infos)
        throughput = _mean_throughput(infos) if feasible else 0.0
        flat_index = environment.flatten_action(list(action))
        selection = ProfileActionSelection(action, throughput, feasible)
        if best is None or throughput > best.mean_throughput_tok_s + 1e-12:
            best = selection
            best_flat_index = flat_index
        elif (
            best_flat_index is not None
            and abs(throughput - best.mean_throughput_tok_s) <= 1e-12
            and flat_index < best_flat_index
        ):
            best = selection
            best_flat_index = flat_index

    if best is None:
        raise ValueError("Profile selection needs at least one candidate action")
    return best


def _strict_local_candidates(environment: DeepFlowEnv) -> List[List[int]]:
    return [[mb_idx, 0, environment.num_layers] for mb_idx in range(len(environment.mb_options))]


def _remote_candidates(environment: DeepFlowEnv) -> List[List[int]]:
    return [[mb_idx, 0, 0] for mb_idx in range(len(environment.mb_options))]


def _legacy_candidates(environment: DeepFlowEnv) -> List[List[int]]:
    return [
        [mb_idx, 0, partition_point]
        for mb_idx in range(len(environment.mb_options))
        for partition_point in range(1, environment.num_layers)
    ]


def _all_candidates(environment: DeepFlowEnv) -> List[List[int]]:
    return [environment.unflatten_action(index) for index in range(environment.num_discrete_actions)]


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _summarize_policy_trials(
    infos: Sequence[Mapping[str, Any]],
    oracle_mean_throughput: float,
) -> Dict[str, Any]:
    throughputs = [float(info["throughput"]) for info in infos]
    makespans = [float(info["makespan"]) for info in infos]
    feasible = [bool(info["feasible"]) for info in infos]
    oom_distribution = Counter(str(info["oom_device"]) for info in infos if not bool(info["feasible"]))
    avg_throughput = float(mean(throughputs)) if throughputs else 0.0
    return {
        "trial_count": len(infos),
        "mean_throughput_tok_s": avg_throughput,
        "p5_throughput_tok_s": _quantile(throughputs, 0.05),
        "p95_makespan_s": _quantile(makespans, 0.95),
        "feasible_rate": float(mean(1.0 if value else 0.0 for value in feasible)) if feasible else 0.0,
        "oom_distribution": dict(sorted(oom_distribution.items())),
        "oracle_ratio": avg_throughput / oracle_mean_throughput if oracle_mean_throughput > 0.0 else 0.0,
    }


def _write_summary(
    output_dir: str,
    suite_name: str,
    metadata: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
) -> Tuple[Path, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / f"{suite_name}_summary.json"
    csv_path = root / f"{suite_name}_summary.csv"
    payload = {"metadata": dict(metadata), "summaries": list(summaries)}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fieldnames = (
        "suite",
        "scenario_id",
        "pressure_profile",
        "policy_name",
        "action_mb_idx",
        "action_k_idx",
        "action_partition_point",
        "trial_count",
        "mean_throughput_tok_s",
        "p5_throughput_tok_s",
        "p95_makespan_s",
        "feasible_rate",
        "oracle_ratio",
        "oom_distribution_json",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row = dict(summary)
            row["oom_distribution_json"] = json.dumps(row.pop("oom_distribution"), sort_keys=True)
            writer.writerow(row)
    return json_path, csv_path


def run_stress_experiments(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    selected_profiles = [get_stress_profile(name) for name in args.profiles]
    scenarios = STRESS_SCENARIOS[:args.max_scenarios] if args.max_scenarios else STRESS_SCENARIOS
    if not scenarios:
        raise ValueError("At least one stress scenario is required")

    model: Optional[PPO] = None
    vec_env: Optional[VecNormalize] = None
    ppo_model_path = None
    vec_normalize_path = None
    if not args.skip_ppo:
        model, vec_env, ppo_model_path, vec_normalize_path = load_best_agent(
            model_path=args.model_path,
            vec_normalize_path=args.vec_normalize_path,
            config_dir=args.config_dir,
        )
        raw_env = vec_env.envs[0]
    else:
        raw_env = DeepFlowEnv(
            config_dir=args.config_dir,
            total_batch_size=32,
            episode_len=1,
            domain_randomization=False,
            seed=args.random_seed,
            reward_mode="shaped",
        )

    metadata = build_run_metadata(
        suite=args.suite_name,
        config_dir=raw_env.config_dir,
        total_batch_size=raw_env.total_batch_size,
        pressure_profile="mixed",
        random_seed=args.random_seed,
        ppo_model_path=ppo_model_path,
        vec_normalize_path=vec_normalize_path,
    )
    metadata.update({
        "stress_profiles": [profile.as_metadata() for profile in selected_profiles],
        "stress_scenarios": [
            {
                "scenario_id": scenario_id,
                "bandwidth_mbps": bandwidth_mbps,
                "link_delay_ms": link_delay_ms,
                "prompt_len": prompt_len,
            }
            for scenario_id, bandwidth_mbps, link_delay_ms, prompt_len in scenarios
        ],
        "trial_count": args.trials,
        "common_random_numbers": {
            "sampling": "max(0, Normal(0, jitter_std)) plus one Bernoulli-triggered generic retry",
            "master_seed": args.random_seed,
            "transmissions_per_trial": raw_env.total_batch_size,
        },
        "ppo_included": not args.skip_ppo,
    })
    recorder = ResultRecorder(args.output_dir, metadata)
    global_static: GlobalStaticSelection = select_global_static_deepflow(raw_env)
    metadata["global_static_deepflow"] = global_static.as_metadata()
    recorder.update_metadata(global_static_deepflow=global_static.as_metadata())

    summaries: List[Dict[str, Any]] = []
    for profile in selected_profiles:
        for scenario_id, bandwidth_mbps, link_delay_ms, prompt_len in scenarios:
            _set_scenario(raw_env, bandwidth_mbps, link_delay_ms, prompt_len)
            scenario_seed = _stable_scenario_seed(args.random_seed, profile.name, scenario_id)
            network_trials = generate_network_trials(
                profile,
                random_seed=scenario_seed,
                trial_count=args.trials,
                transmissions_per_trial=raw_env.total_batch_size,
            )

            strict_local = select_profile_expected_action(
                raw_env, _strict_local_candidates(raw_env), profile, network_trials,
            )
            remote_target = select_profile_expected_action(
                raw_env, _remote_candidates(raw_env), profile, network_trials,
            )
            legacy = select_profile_expected_action(
                raw_env, _legacy_candidates(raw_env), profile, network_trials,
            )
            profile_oracle = select_profile_expected_action(
                raw_env, _all_candidates(raw_env), profile, network_trials,
            )

            selected_actions: Dict[str, Sequence[int]] = {
                "strict_local_target": strict_local.action,
                "remote_target_without_speculation": remote_target.action,
                "best_feasible_legacy_activation_split": legacy.action,
                "token_speculation_without_pipeline": token_speculation_without_pipeline_action(raw_env),
                "global_static_deepflow": global_static.action,
                "heuristic_deepflow": heuristic_deepflow_action(raw_env, link_delay_ms, prompt_len),
                "per_profile_oracle": profile_oracle.action,
            }
            if model is not None and vec_env is not None:
                selected_actions["ppo_deepflow"] = _predict_best_action(model, vec_env, raw_env)

            scenario_metadata = {
                "strict_local_target": strict_local.as_metadata(raw_env),
                "remote_target_without_speculation": remote_target.as_metadata(raw_env),
                "best_feasible_legacy_activation_split": legacy.as_metadata(raw_env),
                "per_profile_oracle": profile_oracle.as_metadata(raw_env),
                "common_random_seed": scenario_seed,
            }
            metadata.setdefault("profile_action_selections", {}).setdefault(profile.name, {})[scenario_id] = scenario_metadata

            policy_infos = {
                policy_name: _trial_infos(raw_env, action, profile, network_trials)
                for policy_name, action in selected_actions.items()
            }
            oracle_mean = _mean_throughput(policy_infos["per_profile_oracle"])
            for policy_name, action in selected_actions.items():
                infos = policy_infos[policy_name]
                for trial_index, info in enumerate(infos):
                    recorder.record(
                        scenario_id=scenario_id,
                        bandwidth_mbps=bandwidth_mbps,
                        link_delay_ms=link_delay_ms,
                        prompt_len=prompt_len,
                        policy_name=policy_name,
                        action=action,
                        info=info,
                        environment=raw_env,
                        pressure_profile=profile.name,
                        random_seed=scenario_seed,
                        trial_index=trial_index,
                    )
                summary = _summarize_policy_trials(infos, oracle_mean)
                summaries.append({
                    "suite": args.suite_name,
                    "scenario_id": scenario_id,
                    "pressure_profile": profile.name,
                    "policy_name": policy_name,
                    "action_mb_idx": int(action[0]),
                    "action_k_idx": int(action[1]),
                    "action_partition_point": int(action[2]),
                    **summary,
                })

    recorder.update_metadata(
        profile_action_selections=metadata.get("profile_action_selections", {}),
    )
    raw_json_path, raw_csv_path = recorder.write()
    summary_json_path, summary_csv_path = _write_summary(args.output_dir, args.suite_name, metadata, summaries)
    if vec_env is not None:
        vec_env.close()
    return raw_json_path, raw_csv_path, summary_json_path, summary_csv_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path")
    parser.add_argument("--vec-normalize-path")
    parser.add_argument("--skip-ppo", action="store_true")
    parser.add_argument("--profiles", nargs="+", choices=tuple(STRESS_PROFILES), default=("base", "mild", "severe"))
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--output-dir", default="results/stress_experiments")
    parser.add_argument("--suite-name", default="stress_experiments")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--random-seed", type=int, default=20260919)
    args = parser.parse_args(argv)
    if args.trials <= 0:
        parser.error("--trials must be positive")
    if args.max_scenarios is not None and args.max_scenarios <= 0:
        parser.error("--max-scenarios must be positive")
    if args.skip_ppo and (args.model_path is not None or args.vec_normalize_path is not None):
        parser.error("--skip-ppo cannot be combined with PPO artifact paths")
    if not args.skip_ppo and (args.model_path is None or args.vec_normalize_path is None):
        parser.error("PPO stress evaluation requires both --model-path and --vec-normalize-path")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    paths = run_stress_experiments(args)
    print(f"Raw stress JSON: {paths[0]}")
    print(f"Raw stress CSV : {paths[1]}")
    print(f"Stress summary JSON: {paths[2]}")
    print(f"Stress summary CSV : {paths[3]}")


if __name__ == "__main__":
    main()
