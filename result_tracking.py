"""Structured, traceable result output for unified simulator experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


RESULT_SCHEMA_VERSION = "1.0"
CONFIG_FILENAMES = (
    "devices_paper.json",
    "llama_1b_paper.json",
    "llama2_7b_paper.json",
)
RECORD_FIELDS = (
    "schema_version",
    "suite",
    "scenario_id",
    "bandwidth_mbps",
    "link_delay_ms",
    "prompt_len",
    "pressure_profile",
    "random_seed",
    "trial_index",
    "policy_name",
    "mode",
    "action_mb_idx",
    "action_k_idx",
    "action_partition_point",
    "micro_batch_size",
    "k_steps",
    "valid",
    "feasible",
    "infeasibility_reason",
    "oom_device",
    "throughput_tok_s",
    "makespan_s",
    "data_size_mb",
    "effective_tokens_per_seq",
    "total_effective_tokens",
    "stage_edge_s",
    "stage_comm_s",
    "stage_cloud_s",
    "verify_seq_len",
    "edge_peak_memory_mb",
    "cloud_peak_memory_mb",
    "edge_budget_mb",
    "cloud_budget_mb",
    "config_hash",
    "ppo_model_path",
    "vec_normalize_path",
)


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_metadata(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None

    artifact = Path(path)
    digest = _sha256_file(artifact)
    return {
        "path": artifact.as_posix(),
        "exists": artifact.is_file(),
        "sha256": digest,
    }


def _git_revision() -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def build_run_metadata(
    *,
    suite: str,
    config_dir: str,
    total_batch_size: int,
    pressure_profile: str = "base",
    random_seed: Optional[int] = None,
    ppo_model_path: Optional[str] = None,
    vec_normalize_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Fingerprint simulator inputs and optional PPO artifacts for one suite."""
    config_root = Path(config_dir)
    config_files = []
    for filename in CONFIG_FILENAMES:
        path = config_root / filename
        config_files.append({
            "path": path.as_posix(),
            "sha256": _sha256_file(path),
        })

    config_hash = hashlib.sha256(
        json.dumps(config_files, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "suite": suite,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "config_files": config_files,
        "config_hash": config_hash,
        "total_batch_size": int(total_batch_size),
        "pressure_profile": pressure_profile,
        "random_seed": random_seed,
        "ppo_model": _artifact_metadata(ppo_model_path),
        "vec_normalize": _artifact_metadata(vec_normalize_path),
    }


class ResultRecorder:
    """Collect simulator outcomes and write matching JSON and flat CSV artifacts."""

    def __init__(self, output_dir: str, metadata: Mapping[str, Any]):
        self.output_dir = Path(output_dir)
        self.metadata = dict(metadata)
        self.records: List[Dict[str, Any]] = []

    def record(
        self,
        *,
        scenario_id: str,
        bandwidth_mbps: float,
        link_delay_ms: float,
        prompt_len: int,
        policy_name: str,
        action: Sequence[int],
        info: Mapping[str, Any],
        environment: Any,
        pressure_profile: Optional[str] = None,
        random_seed: Optional[int] = None,
        trial_index: Optional[int] = None,
    ) -> None:
        if len(action) != 3:
            raise ValueError("action must use [mb_idx, k_idx, partition_point] order")

        mb_idx, k_idx, partition_point = (int(value) for value in action)
        stage_costs = info.get("stage_costs", {})
        mode = info.get("mode") or environment.simulator.describe_execution_mode(
            environment.k_options[k_idx], partition_point
        )

        self.records.append({
            "schema_version": RESULT_SCHEMA_VERSION,
            "suite": self.metadata["suite"],
            "scenario_id": scenario_id,
            "bandwidth_mbps": float(bandwidth_mbps),
            "link_delay_ms": float(link_delay_ms),
            "prompt_len": int(prompt_len),
            "pressure_profile": pressure_profile or self.metadata["pressure_profile"],
            "random_seed": self.metadata["random_seed"] if random_seed is None else random_seed,
            "trial_index": trial_index,
            "policy_name": policy_name,
            "mode": str(mode),
            "action_mb_idx": mb_idx,
            "action_k_idx": k_idx,
            "action_partition_point": partition_point,
            "micro_batch_size": int(info.get("micro_batch_size", environment.mb_options[mb_idx])),
            "k_steps": int(info.get("k_steps", environment.k_options[k_idx])),
            "valid": bool(info["valid"]),
            "feasible": bool(info["feasible"]),
            "infeasibility_reason": str(info.get("infeasibility_reason", "none")),
            "oom_device": str(info["oom_device"]),
            "throughput_tok_s": float(info["throughput"]),
            "makespan_s": float(info["makespan"]),
            "data_size_mb": float(info["data_size_mb"]),
            "effective_tokens_per_seq": float(info.get("effective_tokens_per_seq", 0.0)),
            "total_effective_tokens": float(info.get("total_effective_tokens", 0.0)),
            "stage_edge_s": float(stage_costs.get("edge", 0.0)),
            "stage_comm_s": float(stage_costs.get("comm", 0.0)),
            "stage_cloud_s": float(stage_costs.get("cloud", 0.0)),
            "verify_seq_len": int(info.get("verify_seq_len", 0)),
            "edge_peak_memory_mb": float(info["edge_peak_memory_mb"]),
            "cloud_peak_memory_mb": float(info["cloud_peak_memory_mb"]),
            "edge_budget_mb": float(info["edge_budget_mb"]),
            "cloud_budget_mb": float(info["cloud_budget_mb"]),
            "config_hash": self.metadata["config_hash"],
            "ppo_model_path": self._artifact_path("ppo_model"),
            "vec_normalize_path": self._artifact_path("vec_normalize"),
        })

    def _artifact_path(self, key: str) -> str:
        artifact = self.metadata.get(key)
        return "" if artifact is None else str(artifact["path"])

    def update_metadata(self, **values: Any) -> None:
        """Add policy-calibration facts before the suite is written."""
        self.metadata.update(values)

    def write(self) -> Tuple[Path, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        suite = str(self.metadata["suite"])
        json_path = self.output_dir / f"{suite}.json"
        csv_path = self.output_dir / f"{suite}.csv"

        payload = {
            "metadata": self.metadata,
            "records": self.records,
        }
        with json_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")

        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RECORD_FIELDS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(self.records)

        return json_path, csv_path


def required_record_fields() -> Iterable[str]:
    """Expose the flat schema for chart and analysis readers without duplicating columns."""
    return RECORD_FIELDS
