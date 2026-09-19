"""Regression checks for parameterized network and resource stress semantics."""

import json
import math

from core.hardware import Device, NetworkLink
from core.model_spec import LLaMAModel
from engine.simulator import DeepFlowSimulator
from engine.stress import (
    BASE_STRESS_PROFILE,
    MILD_STRESS_PROFILE,
    SEVERE_STRESS_PROFILE,
    NetworkTransmissionSample,
    generate_network_trials,
)


def build_simulator() -> DeepFlowSimulator:
    with open("configs/devices_paper.json", "r", encoding="utf-8") as handle:
        config = json.load(handle)
    sim = config["simulator"]
    return DeepFlowSimulator(
        draft_model=LLaMAModel("configs/llama_1b_paper.json"),
        target_model=LLaMAModel("configs/llama2_7b_paper.json"),
        edge=Device(**config["devices"][0]),
        cloud=Device(**config["devices"][1]),
        network=NetworkLink(**config["network"]),
        memory_budget_ratio=float(sim["memory_budget_ratio"]),
        weight_reservation_factor=float(sim["weight_reservation_factor"]),
        activation_safety_factor=float(sim["activation_safety_factor"]),
        kv_cache_safety_factor=float(sim["kv_cache_safety_factor"]),
        framework_overhead_edge_mb=float(sim["framework_overhead_edge_mb"]),
        framework_overhead_cloud_mb=float(sim["framework_overhead_cloud_mb"]),
        comm_buffer_safety_factor=float(sim["comm_buffer_safety_factor"]),
        verify_workspace_factor=float(sim["verify_workspace_factor"]),
    )


def assert_base_profile_preserves_baseline(simulator: DeepFlowSimulator) -> None:
    baseline = simulator.simulate(
        total_batch_size=32,
        micro_batch_size=1,
        k_steps=1,
        partition_point=0,
        prompt_len=512,
    )
    samples = generate_network_trials(
        BASE_STRESS_PROFILE,
        random_seed=17,
        trial_count=1,
        transmissions_per_trial=32,
    )[0]
    stressed = simulator.simulate(
        total_batch_size=32,
        micro_batch_size=1,
        k_steps=1,
        partition_point=0,
        prompt_len=512,
        stress_profile=BASE_STRESS_PROFILE,
        network_samples=samples,
    )
    assert math.isclose(stressed.makespan, baseline.makespan, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(stressed.throughput, baseline.throughput, rel_tol=0.0, abs_tol=1e-12)
    assert stressed.memory_breakdown["kv_block_size"] == 1.0
    assert stressed.memory_breakdown["fragmentation_reserve"] == 0.0


def assert_network_draws_and_retry_are_reproducible(simulator: DeepFlowSimulator) -> None:
    first = generate_network_trials(
        MILD_STRESS_PROFILE,
        random_seed=20260919,
        trial_count=2,
        transmissions_per_trial=4,
    )
    second = generate_network_trials(
        MILD_STRESS_PROFILE,
        random_seed=20260919,
        trial_count=2,
        transmissions_per_trial=4,
    )
    assert first == second

    action_kwargs = {
        "total_batch_size": 32,
        "micro_batch_size": 32,
        "k_steps": 0,
        "partition_point": 0,
        "prompt_len": 512,
    }
    no_loss = simulator.simulate(
        **action_kwargs,
        stress_profile=BASE_STRESS_PROFILE,
        network_samples=[NetworkTransmissionSample(0.0, 0.0, False)],
    )
    retry = simulator.simulate(
        **action_kwargs,
        stress_profile=BASE_STRESS_PROFILE,
        network_samples=[NetworkTransmissionSample(0.0, 0.0, True)],
    )
    assert math.isclose(
        retry.stage_costs["comm"],
        no_loss.stage_costs["comm"] * 2.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def assert_resource_pressure_changes_unified_results(simulator: DeepFlowSimulator) -> None:
    base_cost = simulator.estimate_stage_costs(
        micro_batch_size=2,
        k_steps=1,
        partition_point=0,
        prompt_len=513,
        stress_profile=BASE_STRESS_PROFILE,
        network_sample=NetworkTransmissionSample(0.0, 0.0, False),
    )
    mild_cost = simulator.estimate_stage_costs(
        micro_batch_size=2,
        k_steps=1,
        partition_point=0,
        prompt_len=513,
        stress_profile=MILD_STRESS_PROFILE,
        network_sample=NetworkTransmissionSample(0.0, 0.0, False),
    )
    assert math.isclose(mild_cost.edge_time, base_cost.edge_time * 1.10, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(mild_cost.cloud_time, base_cost.cloud_time * 1.10, rel_tol=0.0, abs_tol=1e-12)
    assert mild_cost.comm_time > base_cost.comm_time

    base_mem = simulator.estimate_peak_memory(
        micro_batch_size=2,
        k_steps=1,
        partition_point=0,
        prompt_len=513,
        stress_profile=BASE_STRESS_PROFILE,
    )
    severe_mem = simulator.estimate_peak_memory(
        micro_batch_size=2,
        k_steps=1,
        partition_point=0,
        prompt_len=513,
        stress_profile=SEVERE_STRESS_PROFILE,
    )
    assert math.isclose(severe_mem.edge_budget_mb, base_mem.edge_budget_mb * 0.8, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(severe_mem.cloud_budget_mb, base_mem.cloud_budget_mb * 0.8, rel_tol=0.0, abs_tol=1e-12)
    assert severe_mem.breakdown["edge_draft_kv_allocated_tokens"] > 514.0
    assert severe_mem.breakdown["cloud_suffix_kv_allocated_tokens"] > 515.0

    strict_local = simulator.simulate(
        total_batch_size=32,
        micro_batch_size=1,
        k_steps=0,
        partition_point=len(simulator.target_model.layers),
        prompt_len=512,
        stress_profile=SEVERE_STRESS_PROFILE,
    )
    assert strict_local.stage_costs["comm"] == 0.0


def assert_batched_trials_match_single_trial_simulation(simulator: DeepFlowSimulator) -> None:
    trials = generate_network_trials(
        MILD_STRESS_PROFILE,
        random_seed=314159,
        trial_count=2,
        transmissions_per_trial=16,
    )
    kwargs = {
        "total_batch_size": 32,
        "micro_batch_size": 2,
        "k_steps": 1,
        "partition_point": 0,
        "prompt_len": 512,
        "stress_profile": MILD_STRESS_PROFILE,
    }
    single_results = [
        simulator.simulate(**kwargs, network_samples=trial)
        for trial in trials
    ]
    batch_results = simulator.simulate_stress_trials(**kwargs, network_trials=trials)
    for single, batch in zip(single_results, batch_results):
        assert math.isclose(batch.makespan, single.makespan, rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(batch.throughput, single.throughput, rel_tol=0.0, abs_tol=1e-12)
        assert batch.stage_costs == single.stage_costs
        assert batch.timeline == []


def main() -> None:
    simulator = build_simulator()
    assert_base_profile_preserves_baseline(simulator)
    assert_network_draws_and_retry_are_reproducible(simulator)
    assert_resource_pressure_changes_unified_results(simulator)
    assert_batched_trials_match_single_trial_simulation(simulator)
    print("Stress simulator regression checks passed.")


if __name__ == "__main__":
    main()
