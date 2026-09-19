"""Focused contract checks for common-random-number stress policy selection."""

from engine.stress import MILD_STRESS_PROFILE, generate_network_trials
from rl.envs.flow_env import DeepFlowEnv
from run_stress_experiments import (
    _legacy_candidates,
    _set_scenario,
    select_profile_expected_action,
)


def main() -> None:
    environment = DeepFlowEnv(config_dir="configs", domain_randomization=False, episode_len=1)
    _set_scenario(environment, bandwidth_mbps=1.0, link_delay_ms=80.0, prompt_len=512)
    trials = generate_network_trials(
        MILD_STRESS_PROFILE,
        random_seed=20260919,
        trial_count=2,
        transmissions_per_trial=environment.total_batch_size,
    )
    candidates = _legacy_candidates(environment)[:8]
    first = select_profile_expected_action(environment, candidates, MILD_STRESS_PROFILE, trials)
    second = select_profile_expected_action(environment, candidates, MILD_STRESS_PROFILE, trials)
    assert first == second
    assert list(first.action) in candidates
    assert first.feasible
    assert first.mean_throughput_tok_s > 0.0
    print("Stress experiment common-random-number regression checks passed.")


if __name__ == "__main__":
    main()
