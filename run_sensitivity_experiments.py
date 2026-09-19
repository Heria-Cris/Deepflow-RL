"""Run target-scale, acceptance-profile, and context-boundary simulator audits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from policy_baselines import (
    GLOBAL_STATIC_CALIBRATION_SCENARIOS,
    INDEPENDENT_TEST_SCENARIOS,
    GlobalStaticSelection,
    heuristic_deepflow_action,
    search_per_scenario_oracle,
    select_global_static_deepflow,
    token_speculation_without_pipeline_action,
)
from result_tracking import ResultRecorder, build_run_metadata
from rl.envs.flow_env import DeepFlowEnv


@dataclass(frozen=True)
class AcceptanceProfile:
    name: str
    alpha_base: float
    decay_factor: float

    def acceptance_fn(self, step_index: int) -> float:
        return self.alpha_base * (self.decay_factor ** (step_index - 1))

    def as_metadata(self) -> Dict[str, float | str]:
        return {
            "name": self.name,
            "alpha_base": self.alpha_base,
            "decay_factor": self.decay_factor,
            "formula": f"{self.alpha_base:g} * {self.decay_factor:g}^(j-1)",
        }


ACCEPTANCE_PROFILES: Dict[str, AcceptanceProfile] = {
    "conservative": AcceptanceProfile("conservative", 0.65, 0.75),
    "default": AcceptanceProfile("default", 0.85, 0.85),
    "favorable": AcceptanceProfile("favorable", 0.90, 0.90),
}
TARGET_MODEL_CONFIGS = {
    "7b": "llama2_7b_paper.json",
    "13b": "llama2_13b_paper.json",
}
PROTOCOL_POLICY_NAMES = (
    "remote_target_without_speculation",
    "best_feasible_legacy_activation_split",
    "token_speculation_without_pipeline",
    "global_static_deepflow",
    "heuristic_deepflow",
    "per_scenario_oracle",
)
CONTEXT_PROMPTS = (2048, 4096, 8192)


def _set_scenario(environment: DeepFlowEnv, bandwidth_mbps: float, link_delay_ms: float, prompt_len: int) -> None:
    environment.reset(options={
        "bandwidth_mbps": bandwidth_mbps,
        "latency_ms": link_delay_ms,
        "prompt_len": prompt_len,
    })
    environment.set_scenario(bandwidth_mbps, link_delay_ms, prompt_len)


def _select_best_action(
    environment: DeepFlowEnv,
    candidates: Iterable[Sequence[int]],
    acceptance_fn: Callable[[int], float],
) -> Tuple[List[int], Dict[str, Any]]:
    best_action: Optional[List[int]] = None
    best_info: Optional[Dict[str, Any]] = None
    best_throughput = -1.0
    best_flat_index: Optional[int] = None
    for candidate in candidates:
        action = [int(value) for value in candidate]
        info = environment.evaluate_action(action, acceptance_fn=acceptance_fn)
        if not (info["valid"] and info["feasible"]):
            continue
        throughput = float(info["throughput"])
        flat_index = environment.flatten_action(action)
        if (
            best_action is None
            or throughput > best_throughput + 1e-12
            or (
                abs(throughput - best_throughput) <= 1e-12
                and best_flat_index is not None
                and flat_index < best_flat_index
            )
        ):
            best_action = action
            best_info = info
            best_throughput = throughput
            best_flat_index = flat_index
    if best_action is None or best_info is None:
        raise ValueError("Candidate set has no feasible action in the current scenario")
    return best_action, best_info


def _remote_candidates(environment: DeepFlowEnv) -> List[List[int]]:
    return [[mb_idx, 0, 0] for mb_idx in range(len(environment.mb_options))]


def _legacy_candidates(environment: DeepFlowEnv) -> List[List[int]]:
    return [
        [mb_idx, 0, partition_point]
        for mb_idx in range(len(environment.mb_options))
        for partition_point in range(1, environment.num_layers)
    ]


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    return float(mean(materialized)) if materialized else 0.0


def _write_protocol_summary(
    output_dir: str,
    metadata: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
) -> Tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "sensitivity_protocol_summary.json"
    csv_path = destination / "sensitivity_protocol_summary.csv"
    json_path.write_text(
        json.dumps({"metadata": dict(metadata), "summaries": list(summaries)}, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    fieldnames = (
        "target_model_key",
        "target_model_config",
        "acceptance_profile",
        "policy_name",
        "scenario_count",
        "mean_throughput_tok_s",
        "feasible_rate",
        "mean_oracle_ratio",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    return json_path, csv_path


def _write_context_audit(
    output_dir: str,
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "context_boundary_audit.json"
    csv_path = destination / "context_boundary_audit.csv"
    payload = {
        "metadata": {
            "prompts": list(CONTEXT_PROMPTS),
            "scenario": {"bandwidth_mbps": 1.0, "link_delay_ms": 50.0},
            "scope": "position limits, memory feasibility, and action-domain counts only",
        },
        "rows": list(rows),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = (
        "target_model_key", "target_model_config", "config_hash", "prompt_len", "total_actions",
        "feasible_actions", "feasible_rate", "position_rejections", "draft_position_rejections",
        "target_position_rejections", "combined_position_rejections", "edge_oom_rejections",
        "cloud_oom_rejections", "both_oom_rejections", "other_infeasible_rejections",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def run_context_audit(
    *,
    config_dir: str,
    output_dir: str,
    target_model_keys: Sequence[str],
    random_seed: int,
) -> Tuple[Path, Path]:
    rows: List[Dict[str, Any]] = []
    for target_key in target_model_keys:
        target_config = TARGET_MODEL_CONFIGS[target_key]
        environment = DeepFlowEnv(
            config_dir=config_dir,
            total_batch_size=32,
            episode_len=1,
            domain_randomization=False,
            seed=random_seed,
            reward_mode="shaped",
            target_model_config_filename=target_config,
        )
        metadata = build_run_metadata(
            suite=f"context_audit_{target_key}",
            config_dir=config_dir,
            total_batch_size=environment.total_batch_size,
            random_seed=random_seed,
            config_filenames=("devices_paper.json", "llama_1b_paper.json", target_config),
        )
        for prompt_len in CONTEXT_PROMPTS:
            _set_scenario(environment, 1.0, 50.0, prompt_len)
            counts: Counter[str] = Counter()
            for flat_index in range(environment.num_discrete_actions):
                info = environment.evaluate_action(environment.unflatten_action(flat_index))
                reason = str(info["infeasibility_reason"])
                if info["feasible"]:
                    counts["feasible"] += 1
                elif "position_limit" in reason:
                    counts["position"] += 1
                    if reason == "draft_position_limit":
                        counts["draft_position"] += 1
                    elif reason == "target_position_limit":
                        counts["target_position"] += 1
                    else:
                        counts["combined_position"] += 1
                elif info["oom_device"] in {"edge", "cloud", "both"}:
                    counts[f"{info['oom_device']}_oom"] += 1
                else:
                    counts["other_infeasible"] += 1
            total_actions = environment.num_discrete_actions
            rows.append({
                "target_model_key": target_key,
                "target_model_config": target_config,
                "config_hash": metadata["config_hash"],
                "prompt_len": prompt_len,
                "total_actions": total_actions,
                "feasible_actions": counts["feasible"],
                "feasible_rate": counts["feasible"] / total_actions if total_actions else 0.0,
                "position_rejections": counts["position"],
                "draft_position_rejections": counts["draft_position"],
                "target_position_rejections": counts["target_position"],
                "combined_position_rejections": counts["combined_position"],
                "edge_oom_rejections": counts["edge_oom"],
                "cloud_oom_rejections": counts["cloud_oom"],
                "both_oom_rejections": counts["both_oom"],
                "other_infeasible_rejections": counts["other_infeasible"],
            })
    return _write_context_audit(output_dir, rows)


def run_protocol_comparisons(args: argparse.Namespace) -> Tuple[List[Tuple[Path, Path]], Path, Path]:
    scenarios = (
        INDEPENDENT_TEST_SCENARIOS[:args.max_scenarios]
        if args.max_scenarios is not None
        else INDEPENDENT_TEST_SCENARIOS
    )
    calibration_scenarios = (
        GLOBAL_STATIC_CALIBRATION_SCENARIOS[:args.max_calibration_scenarios]
        if args.max_calibration_scenarios is not None
        else GLOBAL_STATIC_CALIBRATION_SCENARIOS
    )
    if not scenarios or not calibration_scenarios:
        raise ValueError("Both protocol and calibration scenario sets must be non-empty")

    output_paths: List[Tuple[Path, Path]] = []
    summaries: List[Dict[str, Any]] = []
    run_metadata: Dict[str, Any] = {
        "protocol_scenarios": [
            {"scenario_id": scenario_id, "bandwidth_mbps": bandwidth, "link_delay_ms": latency, "prompt_len": prompt}
            for scenario_id, bandwidth, latency, prompt in scenarios
        ],
        "calibration_scenario_count": len(calibration_scenarios),
        "target_models": dict(TARGET_MODEL_CONFIGS),
        "acceptance_profiles": [profile.as_metadata() for profile in ACCEPTANCE_PROFILES.values()],
        "policies": list(PROTOCOL_POLICY_NAMES),
    }
    for target_key in args.target_models:
        target_config = TARGET_MODEL_CONFIGS[target_key]
        environment = DeepFlowEnv(
            config_dir=args.config_dir,
            total_batch_size=32,
            episode_len=1,
            domain_randomization=False,
            seed=args.random_seed,
            reward_mode="shaped",
            target_model_config_filename=target_config,
        )
        for profile_name in args.acceptance_profiles:
            profile = ACCEPTANCE_PROFILES[profile_name]
            acceptance_fn = profile.acceptance_fn
            global_static: GlobalStaticSelection = select_global_static_deepflow(
                environment,
                calibration_scenarios=calibration_scenarios,
                acceptance_fn=acceptance_fn,
            )
            suite_name = f"sensitivity_{target_key}_{profile_name}"
            metadata = build_run_metadata(
                suite=suite_name,
                config_dir=args.config_dir,
                total_batch_size=environment.total_batch_size,
                pressure_profile="base",
                random_seed=args.random_seed,
                config_filenames=("devices_paper.json", "llama_1b_paper.json", target_config),
            )
            metadata.update({
                "target_model_key": target_key,
                "target_model_config": target_config,
                "acceptance_profile": profile_name,
                "acceptance_curve": profile.as_metadata(),
                "global_static_deepflow": global_static.as_metadata(),
                "calibration_scenarios": [
                    {"scenario_id": scenario_id, "bandwidth_mbps": bandwidth, "link_delay_ms": latency, "prompt_len": prompt}
                    for scenario_id, bandwidth, latency, prompt in calibration_scenarios
                ],
                "independent_test_scenarios": run_metadata["protocol_scenarios"],
                "ppo_included": False,
            })
            recorder = ResultRecorder(args.output_dir, metadata)
            by_policy: Dict[str, List[Dict[str, Any]]] = {name: [] for name in PROTOCOL_POLICY_NAMES}
            for scenario_id, bandwidth, latency, prompt_len in scenarios:
                _set_scenario(environment, bandwidth, latency, prompt_len)
                remote_action, remote_info = _select_best_action(environment, _remote_candidates(environment), acceptance_fn)
                legacy_action, legacy_info = _select_best_action(environment, _legacy_candidates(environment), acceptance_fn)
                oracle_action, oracle_info = search_per_scenario_oracle(
                    environment,
                    acceptance_fn=acceptance_fn,
                )
                if oracle_action is None or oracle_info is None:
                    raise ValueError(f"No feasible Oracle action for {scenario_id}")
                selected = {
                    "remote_target_without_speculation": (remote_action, remote_info),
                    "best_feasible_legacy_activation_split": (legacy_action, legacy_info),
                    "token_speculation_without_pipeline": (
                        token_speculation_without_pipeline_action(environment),
                        environment.evaluate_action(
                            token_speculation_without_pipeline_action(environment), acceptance_fn=acceptance_fn
                        ),
                    ),
                    "global_static_deepflow": (
                        list(global_static.action),
                        environment.evaluate_action(list(global_static.action), acceptance_fn=acceptance_fn),
                    ),
                    "heuristic_deepflow": (
                        heuristic_deepflow_action(environment, latency, prompt_len),
                        environment.evaluate_action(
                            heuristic_deepflow_action(environment, latency, prompt_len), acceptance_fn=acceptance_fn
                        ),
                    ),
                    "per_scenario_oracle": (oracle_action, oracle_info),
                }
                for policy_name, (action, info) in selected.items():
                    recorder.record(
                        scenario_id=scenario_id,
                        bandwidth_mbps=bandwidth,
                        link_delay_ms=latency,
                        prompt_len=prompt_len,
                        policy_name=policy_name,
                        action=action,
                        info=info,
                        environment=environment,
                        pressure_profile="base",
                        acceptance_profile=profile_name,
                        target_model_config=target_config,
                        random_seed=args.random_seed,
                    )
                    by_policy[policy_name].append(info)
            output_paths.append(recorder.write())
            oracle_values = by_policy["per_scenario_oracle"]
            for policy_name, infos in by_policy.items():
                oracle_ratios = [
                    float(info["throughput"]) / float(oracle["throughput"])
                    if float(oracle["throughput"]) > 0.0 else 0.0
                    for info, oracle in zip(infos, oracle_values)
                ]
                summaries.append({
                    "target_model_key": target_key,
                    "target_model_config": target_config,
                    "acceptance_profile": profile_name,
                    "policy_name": policy_name,
                    "scenario_count": len(infos),
                    "mean_throughput_tok_s": _mean(info["throughput"] for info in infos),
                    "feasible_rate": _mean(1.0 if info["feasible"] else 0.0 for info in infos),
                    "mean_oracle_ratio": _mean(oracle_ratios),
                })
    summary_paths = _write_protocol_summary(args.output_dir, run_metadata, summaries)
    return output_paths, *summary_paths


def run_sensitivity_experiments(args: argparse.Namespace) -> Dict[str, Any]:
    raw_paths, protocol_summary_json, protocol_summary_csv = run_protocol_comparisons(args)
    context_paths = None
    if not args.skip_context_audit:
        context_paths = run_context_audit(
            config_dir=args.config_dir,
            output_dir=args.output_dir,
            target_model_keys=args.target_models,
            random_seed=args.random_seed,
        )
    return {
        "raw_paths": [(str(json_path), str(csv_path)) for json_path, csv_path in raw_paths],
        "protocol_summary": (str(protocol_summary_json), str(protocol_summary_csv)),
        "context_audit": None if context_paths is None else tuple(str(path) for path in context_paths),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--output-dir", default="results/sensitivity_experiments")
    parser.add_argument("--target-models", nargs="+", choices=tuple(TARGET_MODEL_CONFIGS), default=tuple(TARGET_MODEL_CONFIGS))
    parser.add_argument("--acceptance-profiles", nargs="+", choices=tuple(ACCEPTANCE_PROFILES), default=tuple(ACCEPTANCE_PROFILES))
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--max-calibration-scenarios", type=int)
    parser.add_argument("--skip-context-audit", action="store_true")
    parser.add_argument("--random-seed", type=int, default=20260919)
    args = parser.parse_args(argv)
    for option_name in ("max_scenarios", "max_calibration_scenarios"):
        value = getattr(args, option_name)
        if value is not None and value <= 0:
            parser.error(f"--{option_name.replace('_', '-')} must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    paths = run_sensitivity_experiments(args)
    print(f"Protocol raw result files: {len(paths['raw_paths'])}")
    print(f"Protocol summary JSON: {paths['protocol_summary'][0]}")
    print(f"Protocol summary CSV : {paths['protocol_summary'][1]}")
    if paths["context_audit"] is not None:
        print(f"Context audit JSON   : {paths['context_audit'][0]}")
        print(f"Context audit CSV    : {paths['context_audit'][1]}")


if __name__ == "__main__":
    main()
