# engine/simulator.py
from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional

from core.hardware import Device, NetworkLink
from core.model_spec import LLaMAModel, TransformerLayerSpec
from engine.physics import PhysicsEngine


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

    # ------------------------------------------------------------------
    # Stage cost
    # ------------------------------------------------------------------
    def estimate_stage_costs(
        self,
        micro_batch_size: int,
        k_steps: int,
        partition_point: int,
        prompt_len: int,
        acceptance_fn: Optional[Callable[[int], float]] = None,
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

        if partition_point == 0:
            transfer_tokens = max(1, k_steps)
            data_size_mb = self.target_model.get_token_id_size_mb(
                batch_size=micro_batch_size,
                seq_len=transfer_tokens,
            )
        else:
            last_edge_layer = self.target_model.layers[partition_point - 1]
            data_size_mb = last_edge_layer.get_activation_memory_mb(
                batch_size=micro_batch_size,
                seq_len=k_steps + 1,
            )

        comm_time = PhysicsEngine.estimate_transmission_latency(self.network, data_size_mb)

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
    ) -> MemoryEstimate:
        verify_seq_len = max(1, prompt_len + k_steps + 1)

        target_layers = self.target_model.layers
        draft_layers = self.draft_model.layers
        prefix_layers = target_layers[:partition_point]
        suffix_layers = target_layers[partition_point:]

        # Communication buffer
        if partition_point == 0:
            transfer_tokens = max(1, k_steps)
            data_size_mb = self.target_model.get_token_id_size_mb(
                batch_size=micro_batch_size,
                seq_len=transfer_tokens,
            )
        else:
            last_edge_layer = target_layers[partition_point - 1]
            data_size_mb = last_edge_layer.get_activation_memory_mb(
                batch_size=micro_batch_size,
                seq_len=k_steps + 1,
            )

        comm_buffer_mb = data_size_mb * self.comm_buffer_safety_factor

        # ---------------- Edge ----------------
        edge_draft_weights_mb = self._model_weight_mb(self.draft_model) * self.weight_reservation_factor
        edge_prefix_weights_mb = self._model_weight_mb(
            self.target_model, 0, partition_point
        ) * self.weight_reservation_factor

        # Draft KV: speculative draft model cache
        edge_draft_kv_per_token_mb = self._sum_kv_cache_increment_mb(draft_layers, micro_batch_size)
        edge_draft_kv_total_mb = (
            edge_draft_kv_per_token_mb
            * max(1, prompt_len + k_steps)
            * self.kv_cache_safety_factor
        )

        # Target prefix KV on edge (important for long prompt / large prefix)
        edge_prefix_kv_per_token_mb = self._sum_kv_cache_increment_mb(prefix_layers, micro_batch_size)
        edge_prefix_kv_total_mb = (
            edge_prefix_kv_per_token_mb
            * verify_seq_len
            * self.kv_cache_safety_factor
        )

        edge_draft_act_mb = self._max_activation_mb(draft_layers, micro_batch_size, 1)
        edge_prefix_act_mb = self._sum_activation_mb(prefix_layers, micro_batch_size, verify_seq_len)

        edge_scratch_mb = self.activation_safety_factor * max(
            edge_draft_act_mb,
            edge_prefix_act_mb,
            comm_buffer_mb,
        )

        edge_peak_memory_mb = (
            self.framework_overhead_edge_mb
            + edge_draft_weights_mb
            + edge_prefix_weights_mb
            + edge_draft_kv_total_mb
            + edge_prefix_kv_total_mb
            + edge_scratch_mb
        )

        # ---------------- Cloud ----------------
        cloud_suffix_weights_mb = self._model_weight_mb(
            self.target_model, partition_point, len(target_layers)
        ) * self.weight_reservation_factor

        # Target suffix KV on cloud during verify
        cloud_suffix_kv_per_token_mb = self._sum_kv_cache_increment_mb(suffix_layers, micro_batch_size)
        cloud_suffix_kv_total_mb = (
            cloud_suffix_kv_per_token_mb
            * verify_seq_len
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
            self.framework_overhead_cloud_mb
            + cloud_suffix_weights_mb
            + cloud_suffix_kv_total_mb
            + cloud_workspace_mb
        )

        edge_budget_mb = self.edge.available_memory_gb * 1024.0 * self.memory_budget_ratio
        cloud_budget_mb = self.cloud.available_memory_gb * 1024.0 * self.memory_budget_ratio

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
            "edge_framework_overhead_mb": self.framework_overhead_edge_mb,
            "edge_draft_weights_mb": edge_draft_weights_mb,
            "edge_prefix_weights_mb": edge_prefix_weights_mb,
            "edge_draft_kv_total_mb": edge_draft_kv_total_mb,
            "edge_prefix_kv_total_mb": edge_prefix_kv_total_mb,
            "edge_scratch_mb": edge_scratch_mb,
            "cloud_framework_overhead_mb": self.framework_overhead_cloud_mb,
            "cloud_suffix_weights_mb": cloud_suffix_weights_mb,
            "cloud_suffix_kv_total_mb": cloud_suffix_kv_total_mb,
            "cloud_workspace_mb": cloud_workspace_mb,
            "comm_buffer_mb": comm_buffer_mb,
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
    def simulate(
        self,
        total_batch_size: int,
        micro_batch_size: int,
        k_steps: int,
        partition_point: int,
        prompt_len: int,
        acceptance_fn: Optional[Callable[[int], float]] = None,
    ) -> SimulationResult:
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

        mem = self.estimate_peak_memory(
            micro_batch_size=micro_batch_size,
            k_steps=k_steps,
            partition_point=partition_point,
            prompt_len=prompt_len,
        )

        cost = self.estimate_stage_costs(
            micro_batch_size=micro_batch_size,
            k_steps=k_steps,
            partition_point=partition_point,
            prompt_len=prompt_len,
            acceptance_fn=acceptance_fn,
        )

        if not mem.feasible:
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
                memory_breakdown=mem.breakdown,
            )

        num_micro_batches = total_batch_size // micro_batch_size
        timeline: List[TaskEvent] = []

        t_edge_free = 0.0
        t_net_free = 0.0
        t_cloud_free = 0.0

        active_time_edge = 0.0
        active_time_cloud = 0.0

        for mb_id in range(num_micro_batches):
            start_edge = t_edge_free
            end_edge = start_edge + cost.edge_time
            if cost.edge_time > 0:
                timeline.append(TaskEvent(
                    task_type="Draft",
                    mb_id=mb_id,
                    device_name=self.edge.name,
                    start_time=start_edge,
                    end_time=end_edge,
                    duration=cost.edge_time,
                ))
            t_edge_free = end_edge
            active_time_edge += cost.edge_time

            start_comm = max(end_edge, t_net_free)
            end_comm = start_comm + cost.comm_time
            timeline.append(TaskEvent(
                task_type="Comm",
                mb_id=mb_id,
                device_name="Network",
                start_time=start_comm,
                end_time=end_comm,
                duration=cost.comm_time,
            ))
            t_net_free = end_comm

            start_cloud = max(end_comm, t_cloud_free)
            end_cloud = start_cloud + cost.cloud_time
            if cost.cloud_time > 0:
                timeline.append(TaskEvent(
                    task_type="Verify",
                    mb_id=mb_id,
                    device_name=self.cloud.name,
                    start_time=start_cloud,
                    end_time=end_cloud,
                    duration=cost.cloud_time,
                ))
            t_cloud_free = end_cloud
            active_time_cloud += cost.cloud_time

        makespan = t_cloud_free
        throughput = 0.0
        util_edge = 0.0
        util_cloud = 0.0
        bubble_rate = 1.0

        total_effective_tokens = total_batch_size * cost.effective_tokens_per_seq

        if makespan > 0:
            throughput = total_effective_tokens / makespan
            util_edge = active_time_edge / makespan
            util_cloud = active_time_cloud / makespan
            bubble_rate = 1.0 - (util_edge + util_cloud) / 2.0

        stage_list = [cost.edge_time, cost.comm_time, cost.cloud_time]
        bottleneck_stage = int(max(range(3), key=lambda i: stage_list[i]))

        return SimulationResult(
            feasible=True,
            makespan=makespan,
            throughput=throughput,
            timeline=timeline,
            util_edge=util_edge,
            util_cloud=util_cloud,
            bubble_rate=bubble_rate,
            bottleneck_stage=bottleneck_stage,
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
            memory_breakdown=mem.breakdown,
        )