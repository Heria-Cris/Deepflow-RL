"""Regression checks for the P0 simulator protocol semantics."""

import json

from core.hardware import Device, NetworkLink
from core.model_spec import LLaMAModel
from engine.simulator import DeepFlowSimulator


def build_simulator() -> DeepFlowSimulator:
    with open("configs/devices_paper.json", "r", encoding="utf-8") as handle:
        config = json.load(handle)

    simulator_config = config.get("simulator", {})
    return DeepFlowSimulator(
        draft_model=LLaMAModel("configs/llama_1b_paper.json"),
        target_model=LLaMAModel("configs/llama2_7b_paper.json"),
        edge=Device(**config["devices"][0]),
        cloud=Device(**config["devices"][1]),
        network=NetworkLink(**config["network"]),
        memory_budget_ratio=float(simulator_config.get("memory_budget_ratio", 0.78)),
        weight_reservation_factor=float(simulator_config.get("weight_reservation_factor", 1.08)),
        activation_safety_factor=float(simulator_config.get("activation_safety_factor", 2.40)),
        kv_cache_safety_factor=float(simulator_config.get("kv_cache_safety_factor", 1.35)),
        framework_overhead_edge_mb=float(simulator_config.get("framework_overhead_edge_mb", 2500.0)),
        framework_overhead_cloud_mb=float(simulator_config.get("framework_overhead_cloud_mb", 6000.0)),
        comm_buffer_safety_factor=float(simulator_config.get("comm_buffer_safety_factor", 2.0)),
        verify_workspace_factor=float(simulator_config.get("verify_workspace_factor", 2.2)),
    )


def assert_strict_local_target(simulator: DeepFlowSimulator) -> None:
    result = simulator.simulate(
        total_batch_size=32,
        micro_batch_size=1,
        k_steps=0,
        partition_point=len(simulator.target_model.layers),
        prompt_len=512,
    )

    assert simulator.describe_execution_mode(0, len(simulator.target_model.layers)) == "Strict Local Target"
    assert result.data_size_mb == 0.0
    assert result.stage_costs["comm"] == 0.0
    assert result.stage_costs["cloud"] == 0.0
    assert result.memory_breakdown["comm_buffer_mb"] == 0.0
    assert result.memory_breakdown["edge_draft_weights_mb"] == 0.0
    assert result.memory_breakdown["cloud_framework_overhead_mb"] == 0.0
    assert all(event.task_type not in {"Comm", "Verify", "Remote Target", "Target Suffix"} for event in result.timeline)


def assert_remote_target_without_speculation(simulator: DeepFlowSimulator) -> None:
    result = simulator.simulate(
        total_batch_size=32,
        micro_batch_size=1,
        k_steps=0,
        partition_point=0,
        prompt_len=512,
    )

    assert simulator.describe_execution_mode(0, 0) == "Remote Target without Speculation"
    assert result.data_size_mb > 0.0
    assert result.stage_costs["comm"] > 0.0
    assert result.stage_costs["cloud"] > 0.0
    assert result.stage_costs["edge"] == 0.0
    assert result.memory_breakdown["edge_framework_overhead_mb"] == 0.0
    assert result.memory_breakdown["edge_draft_weights_mb"] == 0.0
    assert any(event.task_type == "Comm" for event in result.timeline)
    assert any(event.task_type == "Remote Target" for event in result.timeline)


def assert_non_speculative_split_omits_draft_memory(simulator: DeepFlowSimulator) -> None:
    result = simulator.simulate(
        total_batch_size=32,
        micro_batch_size=1,
        k_steps=0,
        partition_point=16,
        prompt_len=512,
    )

    assert simulator.describe_execution_mode(0, 16) == "Legacy Activation Split"
    assert result.memory_breakdown["edge_draft_weights_mb"] == 0.0
    assert result.memory_breakdown["edge_draft_kv_total_mb"] == 0.0


def assert_position_limits(simulator: DeepFlowSimulator) -> None:
    draft_over_limit = simulator.simulate(
        total_batch_size=32,
        micro_batch_size=1,
        k_steps=1,
        partition_point=0,
        prompt_len=2048,
    )
    assert not draft_over_limit.feasible
    assert draft_over_limit.infeasibility_reason == "draft_position_limit"
    assert draft_over_limit.throughput == 0.0

    target_over_limit = simulator.simulate(
        total_batch_size=32,
        micro_batch_size=1,
        k_steps=0,
        partition_point=0,
        prompt_len=4096,
    )
    assert not target_over_limit.feasible
    assert target_over_limit.infeasibility_reason == "target_position_limit"
    assert target_over_limit.throughput == 0.0


def main() -> None:
    simulator = build_simulator()
    assert_strict_local_target(simulator)
    assert_remote_target_without_speculation(simulator)
    assert_non_speculative_split_omits_draft_memory(simulator)
    assert_position_limits(simulator)
    print("P0 simulator semantic regression checks passed.")


if __name__ == "__main__":
    main()
