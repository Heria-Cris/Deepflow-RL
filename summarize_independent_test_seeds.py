"""Aggregate three explicitly paired PPO independent-test result files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


PPO_RETRAIN_SEEDS = (42, 20260912, 20261127)
EXPECTED_POLICY_NAMES = {
    "best_feasible_legacy_activation_split",
    "global_static_deepflow",
    "heuristic_deepflow",
    "per_scenario_oracle",
    "ppo_deepflow",
    "remote_target_without_speculation",
    "strict_local_target",
    "token_speculation_without_pipeline",
}


def _metric_summary(values: Iterable[float]) -> Dict[str, float]:
    materialized = [float(value) for value in values]
    if not materialized:
        return {"mean": 0.0, "sample_std": 0.0}
    return {
        "mean": float(mean(materialized)),
        "sample_std": float(stdev(materialized)) if len(materialized) > 1 else 0.0,
    }


def load_result_file(path: str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload.get("metadata"), dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"Result file does not match the ResultRecorder schema: {path}")
    return payload


def _records_by_policy(payload: Mapping[str, Any]) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in payload["records"]:
        grouped[str(record["policy_name"])].append(record)
    return dict(grouped)


def summarize_seed(payload: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = payload["metadata"]
    records_by_policy = _records_by_policy(payload)
    oracle_rows = {
        str(record["scenario_id"]): record
        for record in records_by_policy.get("per_scenario_oracle", [])
    }
    if not oracle_rows:
        raise ValueError("Independent-test result has no per_scenario_oracle records")

    policies: Dict[str, Dict[str, Any]] = {}
    for policy_name, records in sorted(records_by_policy.items()):
        throughputs = [float(record["throughput_tok_s"]) for record in records]
        feasible = [1.0 if bool(record["feasible"]) else 0.0 for record in records]
        oracle_ratios: List[float] = []
        for record in records:
            oracle = oracle_rows.get(str(record["scenario_id"]))
            if oracle is None:
                raise ValueError(
                    f"Missing Oracle record for scenario {record['scenario_id']} in policy {policy_name}"
                )
            oracle_throughput = float(oracle["throughput_tok_s"])
            if oracle_throughput > 0.0:
                oracle_ratios.append(float(record["throughput_tok_s"]) / oracle_throughput)

        policies[policy_name] = {
            "scenario_count": len(records),
            "throughput_tok_s": _metric_summary(throughputs),
            "feasible_rate": _metric_summary(feasible),
            "oracle_ratio": _metric_summary(oracle_ratios),
        }

    scenario_ids = sorted({str(record["scenario_id"]) for record in payload["records"]})
    return {
        "seed": int(metadata["random_seed"]),
        "config_hash": metadata["config_hash"],
        "scenario_ids": scenario_ids,
        "ppo_model": metadata.get("ppo_model"),
        "vec_normalize": metadata.get("vec_normalize"),
        "policies": policies,
    }


def build_summary(payloads: Sequence[Mapping[str, Any]], input_paths: Sequence[str]) -> Dict[str, Any]:
    per_seed = [summarize_seed(payload) for payload in payloads]
    if len(per_seed) != len(input_paths):
        raise ValueError("Each input path must correspond to exactly one result payload")

    seeds = [entry["seed"] for entry in per_seed]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate random seed in independent-test results: {seeds}")
    config_hashes = {entry["config_hash"] for entry in per_seed}
    if len(config_hashes) != 1:
        raise ValueError(f"Independent-test result config hashes differ: {sorted(config_hashes)}")
    scenario_sets = {tuple(entry["scenario_ids"]) for entry in per_seed}
    if len(scenario_sets) != 1:
        raise ValueError("Independent-test scenario IDs differ between seed results")
    policy_sets = {tuple(sorted(entry["policies"])) for entry in per_seed}
    if len(policy_sets) != 1:
        raise ValueError("Independent-test policy coverage differs between seed results")

    policy_names = sorted(per_seed[0]["policies"])
    aggregate_by_policy: Dict[str, Dict[str, Any]] = {}
    for policy_name in policy_names:
        aggregate_by_policy[policy_name] = {
            "seed_count": len(per_seed),
            "mean_throughput_tok_s": _metric_summary(
                entry["policies"][policy_name]["throughput_tok_s"]["mean"]
                for entry in per_seed
            ),
            "mean_feasible_rate": _metric_summary(
                entry["policies"][policy_name]["feasible_rate"]["mean"]
                for entry in per_seed
            ),
            "mean_oracle_ratio": _metric_summary(
                entry["policies"][policy_name]["oracle_ratio"]["mean"]
                for entry in per_seed
            ),
        }

    return {
        "schema_version": "1.0",
        "metadata": {
            "input_paths": list(input_paths),
            "seeds": sorted(seeds),
            "config_hash": next(iter(config_hashes)),
            "scenario_count": len(per_seed[0]["scenario_ids"]),
            "policy_names": policy_names,
            "standard_deviation": "sample",
        },
        "per_seed": sorted(per_seed, key=lambda entry: entry["seed"]),
        "aggregate_by_policy": aggregate_by_policy,
    }


def validate_expected_suite(summary: Mapping[str, Any]) -> None:
    metadata = summary["metadata"]
    if metadata["seeds"] != list(PPO_RETRAIN_SEEDS):
        raise ValueError(f"Expected seeds {list(PPO_RETRAIN_SEEDS)}, got {metadata['seeds']}")
    if metadata["scenario_count"] != 32:
        raise ValueError(f"Expected 32 independent scenarios, got {metadata['scenario_count']}")
    if set(metadata["policy_names"]) != EXPECTED_POLICY_NAMES:
        raise ValueError("Independent-test policies do not match the declared comparison suite")


def write_summary(summary: Mapping[str, Any], output_dir: str) -> tuple[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "three_seed_summary.json"
    csv_path = destination / "three_seed_summary.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    fieldnames = [
        "row_type",
        "seed",
        "policy_name",
        "scenario_count",
        "throughput_mean_tok_s",
        "throughput_sample_std_tok_s",
        "feasible_rate_mean",
        "feasible_rate_sample_std",
        "oracle_ratio_mean",
        "oracle_ratio_sample_std",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for seed_result in summary["per_seed"]:
            for policy_name, policy_result in sorted(seed_result["policies"].items()):
                writer.writerow({
                    "row_type": "per_seed",
                    "seed": seed_result["seed"],
                    "policy_name": policy_name,
                    "scenario_count": policy_result["scenario_count"],
                    "throughput_mean_tok_s": policy_result["throughput_tok_s"]["mean"],
                    "throughput_sample_std_tok_s": policy_result["throughput_tok_s"]["sample_std"],
                    "feasible_rate_mean": policy_result["feasible_rate"]["mean"],
                    "feasible_rate_sample_std": policy_result["feasible_rate"]["sample_std"],
                    "oracle_ratio_mean": policy_result["oracle_ratio"]["mean"],
                    "oracle_ratio_sample_std": policy_result["oracle_ratio"]["sample_std"],
                })
        for policy_name, policy_result in sorted(summary["aggregate_by_policy"].items()):
            writer.writerow({
                "row_type": "across_seeds",
                "seed": "",
                "policy_name": policy_name,
                "scenario_count": summary["metadata"]["scenario_count"],
                "throughput_mean_tok_s": policy_result["mean_throughput_tok_s"]["mean"],
                "throughput_sample_std_tok_s": policy_result["mean_throughput_tok_s"]["sample_std"],
                "feasible_rate_mean": policy_result["mean_feasible_rate"]["mean"],
                "feasible_rate_sample_std": policy_result["mean_feasible_rate"]["sample_std"],
                "oracle_ratio_mean": policy_result["mean_oracle_ratio"]["mean"],
                "oracle_ratio_sample_std": policy_result["mean_oracle_ratio"]["sample_std"],
            })
    return str(json_path), str(csv_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    default_inputs = [
        f"results/independent_test/seed_{seed}/independent_test_seed_{seed}.json"
        for seed in PPO_RETRAIN_SEEDS
    ]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", dest="input_paths")
    parser.add_argument("--output-dir", default="results/independent_test")
    args = parser.parse_args(argv)
    args.input_paths = args.input_paths or default_inputs
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    payloads = [load_result_file(path) for path in args.input_paths]
    summary = build_summary(payloads, args.input_paths)
    validate_expected_suite(summary)
    json_path, csv_path = write_summary(summary, args.output_dir)
    ppo_summary = summary["aggregate_by_policy"]["ppo_deepflow"]
    print(f"PPO mean throughput: {ppo_summary['mean_throughput_tok_s']['mean']:.6f} tok/s")
    print(f"PPO throughput sample std: {ppo_summary['mean_throughput_tok_s']['sample_std']:.6f} tok/s")
    print(f"PPO mean Oracle ratio: {ppo_summary['mean_oracle_ratio']['mean']:.6f}")
    print(f"PPO Oracle-ratio sample std: {ppo_summary['mean_oracle_ratio']['sample_std']:.6f}")
    print(f"Three-seed JSON summary: {json_path}")
    print(f"Three-seed CSV summary : {csv_path}")


if __name__ == "__main__":
    main()
