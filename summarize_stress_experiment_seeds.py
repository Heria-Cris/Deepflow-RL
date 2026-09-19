"""Validate and aggregate the three PPO-seed stress-suite summary files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from engine.stress import STRESS_PROFILES
from run_stress_experiments import STRESS_SCENARIOS


PPO_RETRAIN_SEEDS = (42, 20260912, 20261127)
EXPECTED_POLICY_NAMES = {
    "best_feasible_legacy_activation_split",
    "global_static_deepflow",
    "heuristic_deepflow",
    "per_profile_oracle",
    "ppo_deepflow",
    "remote_target_without_speculation",
    "strict_local_target",
    "token_speculation_without_pipeline",
}
METRIC_NAMES = (
    "mean_throughput_tok_s",
    "p5_throughput_tok_s",
    "p95_makespan_s",
    "feasible_rate",
    "oracle_ratio",
)


def _metric_summary(values: Iterable[float]) -> Dict[str, float]:
    materialized = [float(value) for value in values]
    if not materialized:
        return {"mean": 0.0, "sample_std": 0.0}
    return {
        "mean": float(mean(materialized)),
        "sample_std": float(stdev(materialized)) if len(materialized) > 1 else 0.0,
    }


def load_summary_file(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload.get("metadata"), dict) or not isinstance(payload.get("summaries"), list):
        raise ValueError(f"Stress summary does not match the expected schema: {path}")
    return payload


def _artifact_seed(metadata: Mapping[str, Any], key: str) -> int:
    artifact = metadata.get(key)
    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("path"), str):
        raise ValueError(f"Stress summary metadata has no {key} artifact path")
    matches = re.findall(r"(?:^|/)seed_(\d+)(?:/|$)", artifact["path"].replace("\\", "/"))
    if len(matches) != 1:
        raise ValueError(f"Cannot derive one PPO seed from {key} path: {artifact['path']}")
    return int(matches[0])


def _index_summaries(payload: Mapping[str, Any]) -> Dict[Tuple[str, str, str], Mapping[str, Any]]:
    indexed: Dict[Tuple[str, str, str], Mapping[str, Any]] = {}
    for summary in payload["summaries"]:
        if not isinstance(summary, Mapping):
            raise ValueError("Stress summary contains a non-object row")
        try:
            key = (
                str(summary["scenario_id"]),
                str(summary["pressure_profile"]),
                str(summary["policy_name"]),
            )
        except KeyError as error:
            raise ValueError(f"Stress summary row is missing {error.args[0]}") from error
        if key in indexed:
            raise ValueError(f"Duplicate stress summary row: {key}")
        missing_metrics = [metric for metric in METRIC_NAMES if metric not in summary]
        if missing_metrics:
            raise ValueError(f"Stress summary row {key} is missing metrics: {missing_metrics}")
        indexed[key] = summary
    return indexed


def summarize_seed(payload: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = payload["metadata"]
    ppo_seed = _artifact_seed(metadata, "ppo_model")
    vec_seed = _artifact_seed(metadata, "vec_normalize")
    if ppo_seed != vec_seed:
        raise ValueError(
            f"PPO model seed {ppo_seed} and VecNormalize seed {vec_seed} do not match"
        )

    indexed = _index_summaries(payload)
    scenario_ids = sorted({key[0] for key in indexed})
    profiles = sorted({key[1] for key in indexed})
    policies = sorted({key[2] for key in indexed})
    profile_policy: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for profile_name in profiles:
        profile_policy[profile_name] = {}
        for policy_name in policies:
            rows = [
                indexed[(scenario_id, profile_name, policy_name)]
                for scenario_id in scenario_ids
                if (scenario_id, profile_name, policy_name) in indexed
            ]
            if len(rows) != len(scenario_ids):
                raise ValueError(
                    f"Seed {ppo_seed} is missing {profile_name}/{policy_name} stress scenarios"
                )
            profile_policy[profile_name][policy_name] = {
                "scenario_count": len(rows),
                **{metric: float(mean(float(row[metric]) for row in rows)) for metric in METRIC_NAMES},
            }

    return {
        "seed": ppo_seed,
        "config_hash": metadata.get("config_hash"),
        "random_seed": metadata.get("random_seed"),
        "trial_count": metadata.get("trial_count"),
        "scenario_ids": scenario_ids,
        "pressure_profiles": profiles,
        "policy_names": policies,
        "ppo_model": metadata.get("ppo_model"),
        "vec_normalize": metadata.get("vec_normalize"),
        "profile_policy": profile_policy,
    }


def build_summary(payloads: Sequence[Mapping[str, Any]], input_paths: Sequence[str]) -> Dict[str, Any]:
    per_seed = [summarize_seed(payload) for payload in payloads]
    if len(per_seed) != len(input_paths):
        raise ValueError("Each input path must correspond to one stress summary")

    seeds = [entry["seed"] for entry in per_seed]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate PPO seed in stress summaries: {seeds}")
    for field in ("config_hash", "random_seed", "trial_count", "scenario_ids", "pressure_profiles", "policy_names"):
        values = {json.dumps(entry[field], sort_keys=True) for entry in per_seed}
        if len(values) != 1:
            raise ValueError(f"Stress summaries differ in {field}")

    profiles = per_seed[0]["pressure_profiles"]
    policies = per_seed[0]["policy_names"]
    aggregate: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for profile_name in profiles:
        aggregate[profile_name] = {}
        for policy_name in policies:
            aggregate[profile_name][policy_name] = {
                "seed_count": len(per_seed),
                **{
                    metric: _metric_summary(
                        entry["profile_policy"][profile_name][policy_name][metric]
                        for entry in per_seed
                    )
                    for metric in METRIC_NAMES
                },
            }

    return {
        "schema_version": "1.0",
        "metadata": {
            "input_paths": list(input_paths),
            "seeds": sorted(seeds),
            "config_hash": per_seed[0]["config_hash"],
            "common_random_seed": per_seed[0]["random_seed"],
            "trial_count": per_seed[0]["trial_count"],
            "scenario_count": len(per_seed[0]["scenario_ids"]),
            "pressure_profiles": profiles,
            "policy_names": policies,
            "standard_deviation": "sample_across_seed_means",
        },
        "per_seed": sorted(per_seed, key=lambda entry: entry["seed"]),
        "aggregate_by_profile_policy": aggregate,
    }


def validate_expected_suite(summary: Mapping[str, Any]) -> None:
    metadata = summary["metadata"]
    if metadata["seeds"] != list(PPO_RETRAIN_SEEDS):
        raise ValueError(f"Expected PPO seeds {list(PPO_RETRAIN_SEEDS)}, got {metadata['seeds']}")
    if metadata["trial_count"] != 100:
        raise ValueError(f"Expected 100 trials per stress scenario, got {metadata['trial_count']}")
    if metadata["scenario_count"] != len(STRESS_SCENARIOS):
        raise ValueError(
            f"Expected {len(STRESS_SCENARIOS)} stress scenarios, got {metadata['scenario_count']}"
        )
    if set(metadata["pressure_profiles"]) != set(STRESS_PROFILES):
        raise ValueError("Stress pressure profiles do not match the declared Base/Mild/Severe suite")
    if set(metadata["policy_names"]) != EXPECTED_POLICY_NAMES:
        raise ValueError("Stress policies do not match the declared comparison suite")


def write_summary(summary: Mapping[str, Any], output_dir: str) -> Tuple[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "three_seed_stress_summary.json"
    csv_path = destination / "three_seed_stress_summary.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    fieldnames = [
        "row_type", "seed", "pressure_profile", "policy_name", "scenario_count",
        "mean_throughput_tok_s", "mean_throughput_sample_std_tok_s",
        "p5_throughput_tok_s", "p5_throughput_sample_std_tok_s",
        "p95_makespan_s", "p95_makespan_sample_std_s",
        "feasible_rate", "feasible_rate_sample_std",
        "oracle_ratio", "oracle_ratio_sample_std",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for seed_result in summary["per_seed"]:
            for profile_name, policy_map in sorted(seed_result["profile_policy"].items()):
                for policy_name, metrics in sorted(policy_map.items()):
                    writer.writerow({
                        "row_type": "per_seed",
                        "seed": seed_result["seed"],
                        "pressure_profile": profile_name,
                        "policy_name": policy_name,
                        "scenario_count": metrics["scenario_count"],
                        **{metric: metrics[metric] for metric in METRIC_NAMES},
                        "mean_throughput_sample_std_tok_s": "",
                        "p5_throughput_sample_std_tok_s": "",
                        "p95_makespan_sample_std_s": "",
                        "feasible_rate_sample_std": "",
                        "oracle_ratio_sample_std": "",
                    })
        for profile_name, policy_map in sorted(summary["aggregate_by_profile_policy"].items()):
            for policy_name, metrics in sorted(policy_map.items()):
                writer.writerow({
                    "row_type": "across_seeds",
                    "seed": "",
                    "pressure_profile": profile_name,
                    "policy_name": policy_name,
                    "scenario_count": summary["metadata"]["scenario_count"],
                    **{metric: metrics[metric]["mean"] for metric in METRIC_NAMES},
                    "mean_throughput_sample_std_tok_s": metrics["mean_throughput_tok_s"]["sample_std"],
                    "p5_throughput_sample_std_tok_s": metrics["p5_throughput_tok_s"]["sample_std"],
                    "p95_makespan_sample_std_s": metrics["p95_makespan_s"]["sample_std"],
                    "feasible_rate_sample_std": metrics["feasible_rate"]["sample_std"],
                    "oracle_ratio_sample_std": metrics["oracle_ratio"]["sample_std"],
                })
    return str(json_path), str(csv_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", dest="input_paths")
    parser.add_argument("--output-dir", default="results/stress_experiments")
    args = parser.parse_args(argv)
    args.input_paths = args.input_paths or [
        f"results/stress_experiments/seed_{seed}/stress_experiments_seed_{seed}_summary.json"
        for seed in PPO_RETRAIN_SEEDS
    ]
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    payloads = [load_summary_file(path) for path in args.input_paths]
    summary = build_summary(payloads, args.input_paths)
    validate_expected_suite(summary)
    json_path, csv_path = write_summary(summary, args.output_dir)
    ppo_summary = summary["aggregate_by_profile_policy"]["severe"]["ppo_deepflow"]
    print(f"Severe PPO mean throughput: {ppo_summary['mean_throughput_tok_s']['mean']:.6f} tok/s")
    print(f"Severe PPO throughput sample std: {ppo_summary['mean_throughput_tok_s']['sample_std']:.6f} tok/s")
    print(f"Three-seed stress JSON summary: {json_path}")
    print(f"Three-seed stress CSV summary : {csv_path}")


if __name__ == "__main__":
    main()
