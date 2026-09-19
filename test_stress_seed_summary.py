"""Regression checks for three-seed stress-suite aggregation."""

from summarize_stress_experiment_seeds import build_summary, validate_expected_suite


def _payload(seed: int, throughput_offset: float):
    summaries = []
    for scenario_id in ("stress_a", "stress_b"):
        for pressure_profile in ("base", "mild"):
            for policy_name in ("ppo_deepflow", "per_profile_oracle"):
                throughput = 10.0 + throughput_offset
                if policy_name == "ppo_deepflow":
                    throughput -= 1.0
                summaries.append({
                    "scenario_id": scenario_id,
                    "pressure_profile": pressure_profile,
                    "policy_name": policy_name,
                    "mean_throughput_tok_s": throughput,
                    "p5_throughput_tok_s": throughput - 0.5,
                    "p95_makespan_s": 2.0 + throughput_offset,
                    "feasible_rate": 1.0,
                    "oracle_ratio": throughput / (10.0 + throughput_offset),
                })
    return {
        "metadata": {
            "config_hash": "config-hash",
            "random_seed": 20260919,
            "trial_count": 2,
            "ppo_model": {"path": f"models/ppo_deepflow/seeds/seed_{seed}/best_model/best_model.zip"},
            "vec_normalize": {"path": f"models/ppo_deepflow/seeds/seed_{seed}/best_model/vec_normalize.pkl"},
        },
        "summaries": summaries,
    }


def main() -> None:
    summary = build_summary([
        _payload(42, 0.0),
        _payload(20260912, 2.0),
        _payload(20261127, 4.0),
    ], ["seed_42.json", "seed_20260912.json", "seed_20261127.json"])

    ppo = summary["aggregate_by_profile_policy"]["base"]["ppo_deepflow"]
    assert ppo["mean_throughput_tok_s"]["mean"] == 11.0
    assert ppo["mean_throughput_tok_s"]["sample_std"] == 2.0
    assert ppo["oracle_ratio"]["mean"] < 1.0
    assert summary["metadata"]["scenario_count"] == 2

    try:
        validate_expected_suite(summary)
    except ValueError:
        pass
    else:
        raise AssertionError("Synthetic fixture must not satisfy the full stress-suite contract")

    mismatched = _payload(20261127, 4.0)
    mismatched["metadata"]["random_seed"] = 123
    try:
        build_summary([_payload(42, 0.0), mismatched], ["a.json", "b.json"])
    except ValueError as error:
        assert "random_seed" in str(error)
    else:
        raise AssertionError("Different common-random seeds must be rejected")

    print("Stress three-seed aggregation regression checks passed.")


if __name__ == "__main__":
    main()
