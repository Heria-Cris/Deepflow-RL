# engine/simulator.py
from __future__ import annotations

import dataclasses
import math
from typing import Callable, Dict, List, Optional, Sequence

from core.hardware import Device, NetworkLink
from core.model_spec import LLaMAModel, TransformerLayerSpec
from engine.physics import PhysicsEngine
from engine.stress import NetworkTransmissionSample, StressProfile


@dataclasses.dataclass
class TaskEvent:
    task_type: str
    mb_id: int
    device_name: str
    start_time: float
    end_time: float
    duration: float


@dataclasses.dataclass
class StageCost:
    edge_time: float
    comm_time: float
    cloud_time: float
    data_size_mb: float
    effective_tokens_per_seq: float
    verify_seq_len: int


@dataclasses.dataclass
class MemoryEstimate:
    feasible: bool
    edge_peak_memory_mb: float
    cloud_peak_memory_mb: float
    edge_budget_mb: float
    cloud_budget_mb: float
    oom_device: str
    breakdown: Dict[str, float]


@dataclasses.dataclass
class SimulationResult:
    feasible: bool
    makespan: float
    throughput: float
    timeline: List[TaskEvent]
    util_edge: float
    util_cloud: float
    bubble_rate: float
    bottleneck_stage: int
    data_size_mb: float
    effective_tokens_per_seq: float
    total_effective_tokens: float
    stage_costs: Dict[str, float]
    verify_seq_len: int
    edge_peak_memory_mb: float
    cloud_peak_memory_mb: float
    edge_budget_mb: float
    cloud_budget_mb: float
    oom_device: str
    infeasibility_reason: str
    memory_breakdown: Dict[str, float]


class DeepFlowSimulator:
    """
    统一 simulator backend (v2):
    - throughput / makespan
    - pipeline timeline
    - stricter memory model with target KV cache
    """

    def __init__(
        self,
        draft_model: LLaMAModel,
        target_model: LLaMAModel,
        edge: Device,
        cloud: Device,
        network: NetworkLink,
        memory_budget_ratio: float = 0.78,
        weight_reservation_factor: float = 1.08,
        activation_safety_factor: float = 2.40,
        kv_cache_safety_factor: float = 1.35,
        framework_overhead_edge_mb: float = 2500.0,
        framework_overhead_cloud_mb: float = 6000.0,
        comm_buffer_safety_factor: float = 2.0,
        verify_workspace_factor: float = 2.2,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.edge = edge
        self.cloud = cloud
        self.network = network

        self.memory_budget_ratio = memory_budget_ratio
        self.weight_reservation_factor = weight_reservation_factor
        self.activation_safety_factor = activation_safety_factor
        self.kv_cache_safety_factor = kv_cache_safety_factor
        self.framework_overhead_edge_mb = framework_overhead_edge_mb
        self.framework_overhead_cloud_mb = framework_overhead_cloud_mb
        self.comm_buffer_safety_factor = comm_buffer_safety_factor
        self.verify_workspace_factor = verify_workspace_factor

    # ------------------------------------------------------------------
    # Acceptance / Yield
    # ------------------------------------------------------------------
    @staticmethod
    def default_acceptance_fn(step_index: int) -> float:
        alpha_base = 0.85
        decay_factor = 0.85
        return alpha_base * (decay_factor ** (step_index - 1))

    @staticmethod
    def expected_accepted_tokens(
        k_steps: int,
        acceptance_fn: Optional[Callable[[int], float]] = None,
    ) -> float:
        if acceptance_fn is None:
            acceptance_fn = DeepFlowSimulator.default_acceptance_fn

        if k_steps <= 0:
            return 1.0

        expected_spec = 0.0
        for i in range(1, k_steps + 1):
            expected_spec += acceptance_fn(i)
        return 1.0 + expected_spec

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _layer_weight_mb(layer: TransformerLayerSpec) -> float:
        return (layer.get_parameter_count() * layer.bytes_per_param) / (1024 ** 2)

    @staticmethod
    def _model_weight_mb(model: LLaMAModel, start: int = 0, end: Optional[int] = None) -> float:
        layers = model.layers[start:end]
        return sum(DeepFlowSimulator._layer_weight_mb(layer) for layer in layers)

    @staticmethod
    def _max_activation_mb(
        layers: List[TransformerLayerSpec],
        batch_size: int,
        seq_len: int,
    ) -> float:
        if not layers:
            return 0.0
        return max(layer.get_activation_memory_mb(batch_size, seq_len) for layer in layers)

    @staticmethod
    def _sum_activation_mb(
        layers: List[TransformerLayerSpec],
        batch_size: int,
        seq_len: int,
    ) -> float:
        if not layers:
            return 0.0
        return sum(layer.get_activation_memory_mb(batch_size, seq_len) for layer in layers)

    @staticmethod
    def _sum_kv_cache_increment_mb(
        layers: List[TransformerLayerSpec],
        batch_size: int,
    ) -> float:
        if not layers:
            return 0.0
        return sum(layer.get_kv_cache_increment_mb(batch_size) for layer in layers)

    @staticmethod
    def _allocated_kv_tokens(token_count: int, stress_profile: Optional[StressProfile]) -> int:
        block_size = 1 if stress_profile is None else stress_profile.kv_block_size
        return int(math.ceil(max(1, token_count) / block_size) * block_size)

    def describe_execution_mode(self, k_steps: int, partition_point: int) -> str:
        """Return the externally visible protocol label for one joint action."""
        num_layers = len(self.target_model.layers)
        if partition_point == num_layers:
            if k_steps == 0:
                return "Strict Local Target"
            return "Local Target with Speculation"
        if partition_point == 0:
            if k_steps == 0:
                return "Remote Target without Speculation"
            return "Token Speculative DeepFlow"
        if k_steps == 0:
            return "Legacy Activation Split"
        return "Activation Split with Speculation"

    def _communication_payload_mb(
        self,
        micro_batch_size: int,
        k_steps: int,
        partition_point: int,
    ) -> float:
        """Return the payload only when a target-model suffix runs in the cloud."""
        num_layers = len(self.target_model.layers)
        if partition_point == num_layers:
            return 0.0
        if partition_point == 0:
            # A remote request still carries a minimum Token-ID request when K=0.
            return self.target_model.get_token_id_size_mb(
                batch_size=micro_batch_size,
                seq_len=max(1, k_steps),
            )

        last_edge_layer = self.target_model.layers[partition_point - 1]
        return last_edge_layer.get_activation_memory_mb(
            batch_size=micro_batch_size,
            seq_len=k_steps + 1,
        )

    def _position_limit_reason(
        self,
        k_steps: int,
        partition_point: int,
        prompt_len: int,
    ) -> str:
        """Check only models that participate in the action's execution path."""
        del partition_point  # The target participates for every supported partition.

        violations: List[str] = []
        if k_steps > 0:
            draft_required_len = prompt_len + k_steps
            draft_cap = int(self.draft_model.config["max_position_embeddings"])
            if draft_required_len > draft_cap:
                violations.append("draft_position_limit")

        target_required_len = max(1, prompt_len + k_steps + 1)
        target_cap = int(self.target_model.config["max_position_embeddings"])
        if target_required_len > target_cap:
            violations.append("target_position_limit")

        if not violations:
            return "none"
        return "_and_".join(violations)

    def _edge_task_type(self, k_steps: int, partition_point: int) -> str:
        num_layers = len(self.target_model.layers)
        components: List[str] = []
        if k_steps > 0:
            components.append("Draft")
        if partition_point == num_layers:
            components.append("Local Target")
        elif partition_point > 0:
            components.append("Target Prefix")
        return " + ".join(components)

    def _cloud_task_type(self, k_steps: int, partition_point: int) -> str:
        if partition_point == 0 and k_steps == 0:
            return "Remote Target"
        if k_steps == 0:
            return "Target Suffix"
        return "Verify"

    @staticmethod
    def _infeasible_result(
        cost: StageCost,
        mem: MemoryEstimate,
        reason: str,
    ) -> SimulationResult:
        return SimulationResult(
            feasible=False,
            makespan=1e9,
            throughput=0.0,
            timeline=[],
            util_edge=0.0,
            util_cloud=0.0,
            bubble_rate=1.0,
            bottleneck_stage=-1,
            data_size_mb=cost.data_size_mb,
            effective_tokens_per_seq=cost.effective_tokens_per_seq,
            total_effective_tokens=0.0,
            stage_costs={"edge": cost.edge_time, "comm": cost.comm_time, "cloud": cost.cloud_time},
            verify_seq_len=cost.verify_seq_len,
            edge_peak_memory_mb=mem.edge_peak_memory_mb,
            cloud_peak_memory_mb=mem.cloud_peak_memory_mb,
            edge_budget_mb=mem.edge_budget_mb,
            cloud_budget_mb=mem.cloud_budget_mb,
            oom_device=mem.oom_device,
            infeasibility_reason=reason,
            memory_breakdown=mem.breakdown,
        )

    # ------------------------------------------------------------------
    # Stage cost
    # ------------------------------------------------------------------
    def _estimate_comm_time(
        self,
        data_size_mb: float,
        stress_profile: Optional[StressProfile],
        network_sample: Optional[NetworkTransmissionSample],
    ) -> float:
        if data_size_mb <= 0.0:
            return 0.0
        if stress_profile is None:
            return PhysicsEngine.estimate_transmission_latency(self.network, data_size_mb)
        if network_sample is None:
            if stress_profile.is_stochastic_network:
                raise ValueError(
                    "Stochastic stress profiles require an explicit network sample for each transmission"
                )
            network_sample = NetworkTransmissionSample(0.0, 0.0, False)
        return self.network.estimate_comm_time(
            data_size_mb,
            protocol_overhead_s=stress_profile.protocol_overhead_s,
            jitter_s=network_sample.jitter_s,
            retry_jitter_s=network_sample.retry_jitter_s,
            packet_lost=network_sample.packet_lost,
        )

    def estimate_stage_costs(
        self,
        micro_batch_size: int,
        k_steps: int,
        partition_point: int,
        prompt_len: int,
        acceptance_fn: Optional[Callable[[int], float]] = None,
        stress_profile: Optional[StressProfile] = None,
        network_sample: Optional[NetworkTransmissionSample] = None,
    ) -> StageCost:
        edge_draft_time = 0.0
        if k_steps > 0:
            for step in range(k_steps):
                decode_context_len = max(1, prompt_len + step)
                for layer in self.draft_model.layers:
                    edge_draft_time += PhysicsEngine.estimate_layer_latency(
                        layer=layer,
                        device=self.edge,
                        batch_size=micro_batch_size,
                        seq_len=decode_context_len,
                        is_decoding=True,
                    )

        verify_seq_len = max(1, prompt_len + k_steps + 1)

        edge_target_prefix_time = 0.0
        for i in range(partition_point):
            layer = self.target_model.layers[i]
            edge_target_prefix_time += PhysicsEngine.estimate_layer_latency(
                layer=layer,
                device=self.edge,
                batch_size=micro_batch_size,
                seq_len=verify_seq_len,
                is_decoding=False,
            )

        edge_time = edge_draft_time + edge_target_prefix_time

        data_size_mb = self._communication_payload_mb(
            micro_batch_size=micro_batch_size,
            k_steps=k_steps,
            partition_point=partition_point,
        )

        comm_time = self._estimate_comm_time(data_size_mb, stress_profile, network_sample)

        cloud_time = 0.0
        for i in range(partition_point, len(self.target_model.layers)):
            layer = self.target_model.layers[i]
            cloud_time += PhysicsEngine.estimate_layer_latency(
                layer=layer,
                device=self.cloud,
                batch_size=micro_batch_size,
                seq_len=verify_seq_len,
                is_decoding=False,
            )

        if stress_profile is not None:
            edge_time *= stress_profile.runtime_slowdown
            cloud_time *= stress_profile.runtime_slowdown

        effective_tokens_per_seq = self.expected_accepted_tokens(k_steps, acceptance_fn)

        return StageCost(
            edge_time=edge_time,
            comm_time=comm_time,
            cloud_time=cloud_time,
            data_size_mb=data_size_mb,
            effective_tokens_per_seq=effective_tokens_per_seq,
            verify_seq_len=verify_seq_len,
        )

    # ------------------------------------------------------------------
    # Memory model v2
    # ------------------------------------------------------------------
    def estimate_peak_memory(
        self,
        micro_batch_size: int,
        k_steps: int,
        partition_point: int,
        prompt_len: int,
        stress_profile: Optional[StressProfile] = None,
    ) -> MemoryEstimate:
        verify_seq_len = max(1, prompt_len + k_steps + 1)

        target_layers = self.target_model.layers
        draft_layers = self.draft_model.layers
        prefix_layers = target_layers[:partition_point]
        suffix_layers = target_layers[partition_point:]

        uses_draft = k_steps > 0
        uses_edge_target = partition_point > 0
        uses_cloud_target = partition_point < len(target_layers)

        # Communication buffers exist only when a cloud-side target suffix receives data.
        data_size_mb = self._communication_payload_mb(
            micro_batch_size=micro_batch_size,
            k_steps=k_steps,
            partition_point=partition_point,
        )

        comm_buffer_mb = data_size_mb * self.comm_buffer_safety_factor

        # ---------------- Edge ----------------
        edge_framework_overhead_mb = self.framework_overhead_edge_mb if (uses_draft or uses_edge_target) else 0.0
        edge_draft_weights_mb = (
            self._model_weight_mb(self.draft_model) * self.weight_reservation_factor
            if uses_draft else 0.0
        )
        edge_prefix_weights_mb = self._model_weight_mb(
            self.target_model, 0, partition_point
        ) * self.weight_reservation_factor

        # Draft KV: speculative draft model cache
        edge_draft_kv_per_token_mb = (
            self._sum_kv_cache_increment_mb(draft_layers, micro_batch_size)
            if uses_draft else 0.0
        )
        edge_draft_kv_tokens = self._allocated_kv_tokens(prompt_len + k_steps, stress_profile)
        edge_draft_kv_total_mb = (
            edge_draft_kv_per_token_mb
            * edge_draft_kv_tokens
            * self.kv_cache_safety_factor
        )

        # Target prefix KV on edge (important for long prompt / large prefix)
        edge_prefix_kv_per_token_mb = self._sum_kv_cache_increment_mb(prefix_layers, micro_batch_size)
        edge_prefix_kv_tokens = self._allocated_kv_tokens(verify_seq_len, stress_profile)
        edge_prefix_kv_total_mb = (
            edge_prefix_kv_per_token_mb
            * edge_prefix_kv_tokens
            * self.kv_cache_safety_factor
        )

        edge_draft_act_mb = self._max_activation_mb(draft_layers, micro_batch_size, 1) if uses_draft else 0.0
        edge_prefix_act_mb = self._sum_activation_mb(prefix_layers, micro_batch_size, verify_seq_len)

        edge_scratch_mb = self.activation_safety_factor * max(
            edge_draft_act_mb,
            edge_prefix_act_mb,
            comm_buffer_mb,
        )

        edge_peak_memory_mb = (
            edge_framework_overhead_mb
            + edge_draft_weights_mb
            + edge_prefix_weights_mb
            + edge_draft_kv_total_mb
            + edge_prefix_kv_total_mb
            + edge_scratch_mb
        )

        # ---------------- Cloud ----------------
        cloud_framework_overhead_mb = self.framework_overhead_cloud_mb if uses_cloud_target else 0.0
        cloud_suffix_weights_mb = self._model_weight_mb(
            self.target_model, partition_point, len(target_layers)
        ) * self.weight_reservation_factor

        # Target suffix KV on cloud during verify
        cloud_suffix_kv_per_token_mb = self._sum_kv_cache_increment_mb(suffix_layers, micro_batch_size)
        cloud_suffix_kv_tokens = self._allocated_kv_tokens(verify_seq_len, stress_profile)
        cloud_suffix_kv_total_mb = (
            cloud_suffix_kv_per_token_mb
            * cloud_suffix_kv_tokens
            * self.kv_cache_safety_factor
        )

        # Verify workspace: much stricter than previous version
        cloud_verify_act_sum_mb = self._sum_activation_mb(suffix_layers, micro_batch_size, verify_seq_len)
        cloud_verify_act_max_mb = self._max_activation_mb(suffix_layers, micro_batch_size, verify_seq_len)

        cloud_workspace_mb = self.verify_workspace_factor * max(
            cloud_verify_act_sum_mb * 0.30,
            cloud_verify_act_max_mb,
            comm_buffer_mb,
        )

        cloud_peak_memory_mb = (
            cloud_framework_overhead_mb
            + cloud_suffix_weights_mb
            + cloud_suffix_kv_total_mb
            + cloud_workspace_mb
        )

        fragmentation_multiplier = 1.0 if stress_profile is None else 1.0 - stress_profile.fragmentation_reserve
        edge_budget_mb = self.edge.available_memory_gb * 1024.0 * self.memory_budget_ratio * fragmentation_multiplier
        cloud_budget_mb = self.cloud.available_memory_gb * 1024.0 * self.memory_budget_ratio * fragmentation_multiplier

        feasible = True
        oom_device = "none"
        if edge_peak_memory_mb > edge_budget_mb and cloud_peak_memory_mb > cloud_budget_mb:
            feasible = False
            oom_device = "both"
        elif edge_peak_memory_mb > edge_budget_mb:
            feasible = False
            oom_device = "edge"
        elif cloud_peak_memory_mb > cloud_budget_mb:
            feasible = False
            oom_device = "cloud"

        breakdown = {
            "edge_framework_overhead_mb": edge_framework_overhead_mb,
            "edge_draft_weights_mb": edge_draft_weights_mb,
            "edge_prefix_weights_mb": edge_prefix_weights_mb,
            "edge_draft_kv_total_mb": edge_draft_kv_total_mb,
            "edge_draft_kv_allocated_tokens": float(edge_draft_kv_tokens),
            "edge_prefix_kv_total_mb": edge_prefix_kv_total_mb,
            "edge_prefix_kv_allocated_tokens": float(edge_prefix_kv_tokens),
            "edge_scratch_mb": edge_scratch_mb,
            "cloud_framework_overhead_mb": cloud_framework_overhead_mb,
            "cloud_suffix_weights_mb": cloud_suffix_weights_mb,
            "cloud_suffix_kv_total_mb": cloud_suffix_kv_total_mb,
            "cloud_suffix_kv_allocated_tokens": float(cloud_suffix_kv_tokens),
            "cloud_workspace_mb": cloud_workspace_mb,
            "comm_buffer_mb": comm_buffer_mb,
            "fragmentation_reserve": 1.0 - fragmentation_multiplier,
            "kv_block_size": float(1 if stress_profile is None else stress_profile.kv_block_size),
        }

        return MemoryEstimate(
            feasible=feasible,
            edge_peak_memory_mb=edge_peak_memory_mb,
            cloud_peak_memory_mb=cloud_peak_memory_mb,
            edge_budget_mb=edge_budget_mb,
            cloud_budget_mb=cloud_budget_mb,
            oom_device=oom_device,
            breakdown=breakdown,
        )

    # ------------------------------------------------------------------
    # Timeline simulation
    # ------------------------------------------------------------------
    def _validate_simulation_inputs(
        self,
        total_batch_size: int,
        micro_batch_size: int,
        partition_point: int,
        prompt_len: int,
    ) -> None:
        if micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        if total_batch_size <= 0:
            raise ValueError("total_batch_size must be positive")
        if total_batch_size % micro_batch_size != 0:
            raise ValueError(
                f"Batch size {total_batch_size} must be divisible by micro batch {micro_batch_size}"
            )
        if not (0 <= partition_point <= len(self.target_model.layers)):
            raise ValueError("partition_point out of range")
        if prompt_len <= 0:
            raise ValueError("prompt_len must be positive")

    @staticmethod
    def _average_stage_cost(stage_costs: Sequence[StageCost]) -> StageCost:
        if not stage_costs:
            raise ValueError("At least one micro-batch stage cost is required")
        count = len(stage_costs)
        return StageCost(
            edge_time=sum(value.edge_time for value in stage_costs) / count,
            comm_time=sum(value.comm_time for value in stage_costs) / count,
            cloud_time=sum(value.cloud_time for value in stage_costs) / count,
            data_size_mb=stage_costs[0].data_size_mb,
            effective_tokens_per_seq=stage_costs[0].effective_tokens_per_seq,
            verify_seq_len=stage_costs[0].verify_seq_len,
        )

    def _simulate_pipeline(
        self,
        *,
        total_batch_size: int,
        k_steps: int,
        partition_point: int,
        mem: MemoryEstimate,
        stage_costs: Sequence[StageCost],
        include_timeline: bool,
    ) -> SimulationResult:
        cost = self._average_stage_cost(stage_costs)
        timeline: List[TaskEvent] = []

        t_edge_free = 0.0
        t_net_free = 0.0
        t_cloud_free = 0.0
        active_time_edge = 0.0
        active_time_cloud = 0.0

        for mb_id, micro_batch_cost in enumerate(stage_costs):
            start_edge = t_edge_free
            end_edge = start_edge + micro_batch_cost.edge_time
            if include_timeline and micro_batch_cost.edge_time > 0:
                timeline.append(TaskEvent(
                    task_type=self._edge_task_type(k_steps, partition_point),
                    mb_id=mb_id,
                    device_name=self.edge.name,
                    start_time=start_edge,
                    end_time=end_edge,
                    duration=micro_batch_cost.edge_time,
                ))
            t_edge_free = end_edge
            active_time_edge += micro_batch_cost.edge_time

            start_comm = max(end_edge, t_net_free)
            end_comm = start_comm + micro_batch_cost.comm_time
            if include_timeline and micro_batch_cost.comm_time > 0:
                timeline.append(TaskEvent(
                    task_type="Comm",
                    mb_id=mb_id,
                    device_name="Network",
                    start_time=start_comm,
                    end_time=end_comm,
                    duration=micro_batch_cost.comm_time,
                ))
            t_net_free = end_comm

            start_cloud = max(end_comm, t_cloud_free)
            end_cloud = start_cloud + micro_batch_cost.cloud_time
            if include_timeline and micro_batch_cost.cloud_time > 0:
                timeline.append(TaskEvent(
                    task_type=self._cloud_task_type(k_steps, partition_point),
                    mb_id=mb_id,
                    device_name=self.cloud.name,
                    start_time=start_cloud,
                    end_time=end_cloud,
                    duration=micro_batch_cost.cloud_time,
                ))
            t_cloud_free = end_cloud
            active_time_cloud += micro_batch_cost.cloud_time

        makespan = t_cloud_free
        total_effective_tokens = total_batch_size * cost.effective_tokens_per_seq
        throughput = total_effective_tokens / makespan if makespan > 0.0 else 0.0
        util_edge = active_time_edge / makespan if makespan > 0.0 else 0.0
        util_cloud = active_time_cloud / makespan if makespan > 0.0 else 0.0
        bubble_rate = 1.0 - (util_edge + util_cloud) / 2.0 if makespan > 0.0 else 1.0
        stage_list = [cost.edge_time, cost.comm_time, cost.cloud_time]

        return SimulationResult(
            feasible=True,
            makespan=makespan,
            throughput=throughput,
            timeline=timeline,
            util_edge=util_edge,
            util_cloud=util_cloud,
            bubble_rate=bubble_rate,
            bottleneck_stage=int(max(range(3), key=lambda index: stage_list[index])),
            data_size_mb=cost.data_size_mb,
            effective_tokens_per_seq=cost.effective_tokens_per_seq,
            total_effective_tokens=total_effective_tokens,
            stage_costs={"edge": cost.edge_time, "comm": cost.comm_time, "cloud": cost.cloud_time},
            verify_seq_len=cost.verify_seq_len,
            edge_peak_memory_mb=mem.edge_peak_memory_mb,
            cloud_peak_memory_mb=mem.cloud_peak_memory_mb,
            edge_budget_mb=mem.edge_budget_mb,
            cloud_budget_mb=mem.cloud_budget_mb,
            oom_device=mem.oom_device,
            infeasibility_reason="none",
            memory_breakdown=mem.breakdown,
        )

    def simulate(
        self,
        total_batch_size: int,
        micro_batch_size: int,
        k_steps: int,
        partition_point: int,
        prompt_len: int,
        acceptance_fn: Optional[Callable[[int], float]] = None,
        stress_profile: Optional[StressProfile] = None,
        network_samples: Optional[Sequence[NetworkTransmissionSample]] = None,
    ) -> SimulationResult:
        self._validate_simulation_inputs(total_batch_size, micro_batch_size, partition_point, prompt_len)

        mem = self.estimate_peak_memory(
            micro_batch_size=micro_batch_size,
            k_steps=k_steps,
            partition_point=partition_point,
            prompt_len=prompt_len,
            stress_profile=stress_profile,
        )

        num_micro_batches = total_batch_size // micro_batch_size
        if network_samples is not None and len(network_samples) != num_micro_batches:
            raise ValueError("network_samples must contain one sample for every micro-batch")
        stage_costs = [
            self.estimate_stage_costs(
                micro_batch_size=micro_batch_size,
                k_steps=k_steps,
                partition_point=partition_point,
                prompt_len=prompt_len,
                acceptance_fn=acceptance_fn,
                stress_profile=stress_profile,
                network_sample=None if network_samples is None else network_samples[mb_id],
            )
            for mb_id in range(num_micro_batches)
        ]
        cost = self._average_stage_cost(stage_costs)

        position_reason = self._position_limit_reason(
            k_steps=k_steps,
            partition_point=partition_point,
            prompt_len=prompt_len,
        )
        if position_reason != "none":
            return self._infeasible_result(cost, mem, position_reason)

        if not mem.feasible:
            return self._infeasible_result(cost, mem, f"memory_{mem.oom_device}_oom")

        return self._simulate_pipeline(
            total_batch_size=total_batch_size,
            k_steps=k_steps,
            partition_point=partition_point,
            mem=mem,
            stage_costs=stage_costs,
            include_timeline=True,
        )

    def simulate_stress_trials(
        self,
        *,
        total_batch_size: int,
        micro_batch_size: int,
        k_steps: int,
        partition_point: int,
        prompt_len: int,
        stress_profile: StressProfile,
        network_trials: Sequence[Sequence[NetworkTransmissionSample]],
        acceptance_fn: Optional[Callable[[int], float]] = None,
    ) -> List[SimulationResult]:
        """Evaluate common random network trials while reusing fixed action costs and memory."""
        self._validate_simulation_inputs(total_batch_size, micro_batch_size, partition_point, prompt_len)
        if not network_trials:
            raise ValueError("network_trials must be non-empty")

        num_micro_batches = total_batch_size // micro_batch_size
        if any(len(trial) != num_micro_batches for trial in network_trials):
            raise ValueError("Every network trial must contain one sample for each micro-batch")

        mem = self.estimate_peak_memory(
            micro_batch_size=micro_batch_size,
            k_steps=k_steps,
            partition_point=partition_point,
            prompt_len=prompt_len,
            stress_profile=stress_profile,
        )
        base_cost = self.estimate_stage_costs(
            micro_batch_size=micro_batch_size,
            k_steps=k_steps,
            partition_point=partition_point,
            prompt_len=prompt_len,
            acceptance_fn=acceptance_fn,
            stress_profile=stress_profile,
            network_sample=NetworkTransmissionSample(0.0, 0.0, False),
        )
        position_reason = self._position_limit_reason(
            k_steps=k_steps,
            partition_point=partition_point,
            prompt_len=prompt_len,
        )
        if position_reason != "none":
            return [self._infeasible_result(base_cost, mem, position_reason) for _ in network_trials]
        if not mem.feasible:
            reason = f"memory_{mem.oom_device}_oom"
            return [self._infeasible_result(base_cost, mem, reason) for _ in network_trials]

        results = []
        for trial in network_trials:
            stage_costs = [
                StageCost(
                    edge_time=base_cost.edge_time,
                    comm_time=self._estimate_comm_time(base_cost.data_size_mb, stress_profile, sample),
                    cloud_time=base_cost.cloud_time,
                    data_size_mb=base_cost.data_size_mb,
                    effective_tokens_per_seq=base_cost.effective_tokens_per_seq,
                    verify_seq_len=base_cost.verify_seq_len,
                )
                for sample in trial
            ]
            results.append(self._simulate_pipeline(
                total_batch_size=total_batch_size,
                k_steps=k_steps,
                partition_point=partition_point,
                mem=mem,
                stage_costs=stage_costs,
                include_timeline=False,
            ))
        return results
