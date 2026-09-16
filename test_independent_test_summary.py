"""Regression checks for three-seed independent-test aggregation."""

from summarize_independent_test_seeds import build_summary, validate_expected_suite


def _payload(seed: int, ppo_throughputs):
    records = []
    for index, ppo_throughput in enumerate(ppo_throughputs, 1):
        scenario_id = f"independent_test_{index}"
        records.extend([
            {
                "scenario_id": scenario_id,
                "policy_name": "ppo_deepflow",
                "throughput_tok_s": ppo_throughput,
                "feasible": True,
            },
            {
                "scenario_id": scenario_id,
                "policy_name": "per_scenario_oracle",
                "throughput_tok_s": 10.0,
                "feasible": True,
            },
        ])
    return {
        "metadata": {
            "random_seed": seed,
            "config_hash": "config-hash",
            "ppo_model": {"path": f"seed_{seed}/model.zip"},
            "vec_normalize": {"path": f"seed_{seed}/vec_normalize.pkl"},
        },
        "records": records,
    }


def main() -> None:
    summary = build_summary([
        _payload(42, [5.0, 10.0]),
        _payload(20260912, [6.0, 8.0]),
        _payload(20261127, [4.0, 6.0]),
    ], ["seed_42.json", "seed_20260912.json", "seed_20261127.json"])

    ppo_per_seed = {entry["seed"]: entry["policies"]["ppo_deepflow"] for entry in summary["per_seed"]}
    assert ppo_per_seed[42]["throughput_tok_s"]["mean"] == 7.5
    assert ppo_per_seed[42]["oracle_ratio"]["mean"] == 0.75
    assert summary["aggregate_by_policy"]["ppo_deepflow"]["mean_throughput_tok_s"]["mean"] == 6.5
    assert summary["aggregate_by_policy"]["ppo_deepflow"]["mean_oracle_ratio"]["mean"] == 0.65

    try:
        validate_expected_suite(summary)
    except ValueError:
        pass
    else:
        raise AssertionError("Synthetic fixture must not satisfy the full production-suite contract")

    print("Independent-test aggregation regression checks passed.")


if __name__ == "__main__":
    main()
