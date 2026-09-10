# engine/pipeline.py
from __future__ import annotations

from typing import Dict, List, Tuple

from core.hardware import Device, NetworkLink
from core.model_spec import LLaMAModel
from engine.simulator import DeepFlowSimulator, TaskEvent


class PipelineScheduler:
    """
    流水线调度器。
    改为统一调用 DeepFlowSimulator，避免与 RL 环境各算各的。
    """

    def __init__(
        self,
        draft_model: LLaMAModel,
        target_model: LLaMAModel,
        edge: Device,
        cloud: Device,
        network: NetworkLink,
    ):
        self.simulator = DeepFlowSimulator(
            draft_model=draft_model,
            target_model=target_model,
            edge=edge,
            cloud=cloud,
            network=network,
        )

        self.edge = edge
        self.cloud = cloud

    def schedule(
        self,
        total_batch_size: int,
        micro_batch_size: int,
        k_steps: int,
        prompt_len: int = 512,
        partition_point: int = 0,
    ) -> Tuple[float, List[TaskEvent], Dict]:
        result = self.simulator.simulate(
            total_batch_size=total_batch_size,
            micro_batch_size=micro_batch_size,
            k_steps=k_steps,
            partition_point=partition_point,
            prompt_len=prompt_len,
        )

        metrics = {
            "makespan": result.makespan,
            "throughput": result.throughput,
            "util_edge": result.util_edge,
            "util_cloud": result.util_cloud,
            "bubble_rate": result.bubble_rate,
            "data_size_mb": result.data_size_mb,
            "effective_tokens_per_seq": result.effective_tokens_per_seq,
            "stage_costs": result.stage_costs,
            "verify_seq_len": result.verify_seq_len,
            "bottleneck": result.bottleneck_stage,
        }
        return result.makespan, result.timeline, metrics

    def print_timeline(self, timeline: List[TaskEvent]):
        print("\n[Pipeline Gantt Chart (First 3 & Last 1 Micro-batches)]")
        print(f"{'MB_ID':<5} | {'Device':<18} | {'Type':<8} | {'Start (s)':<10} | {'End (s)':<10}")
        print("-" * 70)

        if not timeline:
            print("(empty timeline)")
            return

        sorted_events = sorted(timeline, key=lambda x: x.start_time)
        max_mb = max(e.mb_id for e in sorted_events)
        display_events = [e for e in sorted_events if e.mb_id < 3 or e.mb_id == max_mb]

        for e in display_events:
            print(
                f"{e.mb_id:<5} | {e.device_name[:18]:<18} | {e.task_type:<8} | "
                f"{e.start_time:<10.4f} | {e.end_time:<10.4f}"
            )