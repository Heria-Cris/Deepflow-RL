"""Regression checks for target-scale, acceptance, and context-boundary analyses."""

import json
import tempfile
from pathlib import Path

from rl.envs.flow_env import DeepFlowEnv
from run_sensitivity_experiments import (
    ACCEPTANCE_PROFILES,
    parse_args,
    run_sensitivity_experiments,
)


def main() -> None:
    default_env = DeepFlowEnv(config_dir="configs", domain_randomization=False, seed=1)
    target_13b_env = DeepFlowEnv(
        config_dir="configs",
        domain_randomization=False,
        seed=1,
        target_model_config_filename="llama2_13b_paper.json",
    )
    assert default_env.num_layers == 32
    assert default_env.num_discrete_actions == 1188
    assert target_13b_env.num_layers == 40
    assert target_13b_env.num_discrete_actions == 1476
    assert ACCEPTANCE_PROFILES["conservative"].acceptance_fn(1) == 0.65
    assert ACCEPTANCE_PROFILES["default"].acceptance_fn(2) == 0.85 * 0.85
    assert ACCEPTANCE_PROFILES["favorable"].acceptance_fn(2) == 0.90 * 0.90

    with tempfile.TemporaryDirectory() as directory:
        args = parse_args([
            "--output-dir", directory,
            "--target-models", "7b",
            "--acceptance-profiles", "default",
            "--max-scenarios", "1",
            "--max-calibration-scenarios", "1",
        ])
        outputs = run_sensitivity_experiments(args)
        raw_json = Path(outputs["raw_paths"][0][0])
        raw_payload = json.loads(raw_json.read_text(encoding="utf-8"))
        assert raw_payload["metadata"]["target_model_config"] == "llama2_7b_paper.json"
        assert raw_payload["metadata"]["acceptance_profile"] == "default"
        assert len(raw_payload["records"]) == 6
        assert {record["acceptance_profile"] for record in raw_payload["records"]} == {"default"}
        assert {record["target_model_config"] for record in raw_payload["records"]} == {"llama2_7b_paper.json"}

        summary_payload = json.loads(Path(outputs["protocol_summary"][0]).read_text(encoding="utf-8"))
        assert len(summary_payload["summaries"]) == 6
        assert all(row["scenario_count"] == 1 for row in summary_payload["summaries"])

        context_payload = json.loads(Path(outputs["context_audit"][0]).read_text(encoding="utf-8"))
        assert len(context_payload["rows"]) == 3
        assert all(row["total_actions"] == 1188 for row in context_payload["rows"])
        assert context_payload["rows"][-1]["position_rejections"] > 0

    print("Sensitivity experiment regression checks passed.")


if __name__ == "__main__":
    main()
