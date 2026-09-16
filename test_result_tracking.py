"""Regression checks for structured experiment result artifacts."""

import csv
import json
import tempfile
from pathlib import Path

from result_tracking import ResultRecorder, build_run_metadata, required_record_fields


class FakeSimulator:
    @staticmethod
    def describe_execution_mode(k_steps, partition_point):
        if partition_point == 0 and k_steps == 0:
            return "Remote Target without Speculation"
        return "Token Speculative DeepFlow"


class FakeEnvironment:
    mb_options = [1, 2, 4]
    k_options = [0, 1, 3]
    simulator = FakeSimulator()


def main() -> None:
    metadata = build_run_metadata(
        suite="tracker_regression",
        config_dir="configs",
        total_batch_size=32,
        random_seed=123,
        ppo_model_path="models/ppo_deepflow/final_model.zip",
        vec_normalize_path="models/ppo_deepflow/vec_normalize.pkl",
    )
    assert metadata["config_hash"]
    assert len(metadata["config_files"]) == 3

    info = {
        "valid": True,
        "feasible": True,
        "mode": "Token Speculative DeepFlow",
        "infeasibility_reason": "none",
        "oom_device": "none",
        "throughput": 12.5,
        "makespan": 2.56,
        "data_size_mb": 0.0005,
        "effective_tokens_per_seq": 1.85,
        "total_effective_tokens": 59.2,
        "stage_costs": {"edge": 1.2, "comm": 0.05, "cloud": 0.3},
        "verify_seq_len": 514,
        "edge_peak_memory_mb": 1234.0,
        "cloud_peak_memory_mb": 5678.0,
        "edge_budget_mb": 10000.0,
        "cloud_budget_mb": 20000.0,
    }

    with tempfile.TemporaryDirectory() as directory:
        recorder = ResultRecorder(directory, metadata)
        recorder.record(
            scenario_id="weak_1mbps_50ms_512",
            bandwidth_mbps=1.0,
            link_delay_ms=50.0,
            prompt_len=512,
            policy_name="ppo_deepflow",
            action=[1, 1, 0],
            info=info,
            environment=FakeEnvironment(),
        )
        json_path, csv_path = recorder.write()

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["metadata"]["config_hash"] == metadata["config_hash"]
        assert payload["records"][0]["mode"] == "Token Speculative DeepFlow"
        assert payload["records"][0]["micro_batch_size"] == 2
        assert payload["records"][0]["stage_comm_s"] == 0.05

        with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            assert tuple(reader.fieldnames or []) == tuple(required_record_fields())
            row = next(reader)
        assert row["policy_name"] == "ppo_deepflow"
        assert row["config_hash"] == metadata["config_hash"]

    print("Structured result tracking regression checks passed.")


if __name__ == "__main__":
    main()
