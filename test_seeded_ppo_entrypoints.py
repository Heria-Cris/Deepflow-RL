"""Contract tests for seed-specific PPO training and evaluation entrypoints."""

import os
import tempfile
from contextlib import redirect_stderr
from io import StringIO

from run_paper_experiments import parse_args as parse_experiment_args, resolve_agent_artifacts
from train_phase5 import (
    DEFAULT_NUM_ENVS,
    DEFAULT_TOTAL_TIMESTEPS,
    PPO_RETRAIN_SEEDS,
    build_training_paths,
    parse_args as parse_training_args,
)


def assert_training_cli_contract() -> None:
    args = parse_training_args([])
    assert args.seed == 42
    assert args.total_timesteps == DEFAULT_TOTAL_TIMESTEPS
    assert args.num_envs == DEFAULT_NUM_ENVS

    seeded = parse_training_args(["--seed", "20261127", "--total-timesteps", "4096"])
    assert seeded.seed == PPO_RETRAIN_SEEDS[-1]
    assert seeded.total_timesteps == 4096

    paths = build_training_paths(20260912, "temporary/seeds")
    assert paths["run_dir"] == os.path.join("temporary/seeds", "seed_20260912")
    assert paths["final_model_path"] == os.path.join(paths["run_dir"], "final_model")
    assert paths["final_vecnorm_path"] == os.path.join(paths["run_dir"], "vec_normalize.pkl")


def assert_evaluation_cli_contract() -> None:
    args = parse_experiment_args([
        "--model-path", "models/ppo_deepflow/seeds/seed_42/best_model/best_model.zip",
        "--vec-normalize-path", "models/ppo_deepflow/seeds/seed_42/best_model/vec_normalize.pkl",
        "--output-dir", "results/independent_test/seed_42",
        "--suite-name", "independent_test_seed_42",
        "--only-independent-test",
    ])
    assert args.only_independent_test
    assert args.suite_name == "independent_test_seed_42"

    with tempfile.TemporaryDirectory() as directory:
        model_path = os.path.join(directory, "model.zip")
        stats_path = os.path.join(directory, "vec_normalize.pkl")
        open(model_path, "wb").close()
        open(stats_path, "wb").close()
        assert resolve_agent_artifacts(model_path, stats_path) == (model_path, stats_path)

    with redirect_stderr(StringIO()):
        try:
            parse_experiment_args(["--model-path", "model.zip"])
        except SystemExit:
            pass
        else:
            raise AssertionError("Unpaired PPO artifact path must be rejected")


def main() -> None:
    assert_training_cli_contract()
    assert_evaluation_cli_contract()
    print("Seeded PPO entrypoint contract checks passed.")


if __name__ == "__main__":
    main()
