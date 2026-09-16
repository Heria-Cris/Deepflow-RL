"""Regression checks for frozen Global Static, Heuristic, and Oracle policies."""

from policy_baselines import (
    GLOBAL_STATIC_CALIBRATION_SCENARIOS,
    INDEPENDENT_TEST_SCENARIOS,
    heuristic_deepflow_action,
    search_per_scenario_oracle,
    select_global_static_deepflow,
    set_scenario,
    token_deepflow_candidate_actions,
    token_speculation_without_pipeline_action,
)
from rl.envs.flow_env import DeepFlowEnv


def assert_heuristic_contract(environment: DeepFlowEnv) -> None:
    cases = [
        (49.0, 128, 4),
        (50.0, 128, 8),
        (39.0, 512, 1),
        (40.0, 512, 2),
        (80.0, 512, 4),
        (20.0, 1536, 1),
    ]
    for link_delay_ms, prompt_len, expected_micro_batch in cases:
        action = heuristic_deepflow_action(environment, link_delay_ms, prompt_len)
        assert action[2] == 0
        assert environment.k_options[action[1]] == 1
        assert environment.mb_options[action[0]] == expected_micro_batch
        assert environment.action_space.contains(action)


def assert_global_static_selection(environment: DeepFlowEnv) -> None:
    small_calibration = (
        ("unit_weak_short", 0.5, 50.0, 128),
        ("unit_strong_medium", 20.0, 20.0, 768),
    )
    first = select_global_static_deepflow(environment, small_calibration)
    second = select_global_static_deepflow(environment, small_calibration)

    assert first == second
    assert first.calibration_scenario_count == len(small_calibration)
    assert first.action[2] == 0
    assert environment.k_options[first.action[1]] > 0
    assert 0.0 <= first.mean_oracle_ratio <= 1.0 + 1e-12

    candidate_actions = token_deepflow_candidate_actions(environment)
    assert list(first.action) in candidate_actions
    assert len(candidate_actions) == len(environment.mb_options) * (len(environment.k_options) - 1)


def assert_oracle_and_ablation_actions(environment: DeepFlowEnv) -> None:
    set_scenario(environment, bandwidth_mbps=1.0, link_delay_ms=50.0, prompt_len=512)
    oracle_action, oracle_info = search_per_scenario_oracle(environment)
    assert oracle_action is not None
    assert oracle_info is not None
    assert oracle_info["valid"] and oracle_info["feasible"]

    no_pipeline_action = token_speculation_without_pipeline_action(environment)
    assert no_pipeline_action == [environment.mb_options.index(32), environment.k_options.index(1), 0]
    assert environment.action_space.contains(no_pipeline_action)


def main() -> None:
    assert len(GLOBAL_STATIC_CALIBRATION_SCENARIOS) == 140
    assert len(INDEPENDENT_TEST_SCENARIOS) == 32

    environment = DeepFlowEnv(config_dir="configs", domain_randomization=False, episode_len=1)
    assert_heuristic_contract(environment)
    assert_global_static_selection(environment)
    assert_oracle_and_ablation_actions(environment)
    print("Policy baseline regression checks passed.")


if __name__ == "__main__":
    main()
