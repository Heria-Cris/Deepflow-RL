# engine/speculative.py
from __future__ import annotations

from typing import Optional, Callable

from core.hardware import Device, NetworkLink
from core.model_spec import LLaMAModel
from engine.simulator import DeepFlowSimulator, SimulationResult


class SpeculativeScheduler:
    """
    投机推理调度器。
    现在不再自己维护另一套公式，而是直接调用统一 simulator backend。
    """

    def __init__(
        self,
        draft_model: LLaMAModel,
        target_model: LLaMAModel,
        edge_device: Device,
        cloud_device: Device,
        network: NetworkLink,
    ):
        self.simulator = DeepFlowSimulator(
            draft_model=draft_model,
            target_model=target_model,
            edge=edge_device,
            cloud=cloud_device,
            network=network,
        )

        self.current_seq_len = 0.0
        self.generated_tokens = 0.0

    def simulate_step(
        self,
        k_steps: int,
        acceptance_rate: Optional[float] = None,
        batch_size: int = 1,
        prompt_len: int = 512,
    ):
        """
        单步 speculative loop 的兼容接口。
        如果传入 acceptance_rate，则构造常数 acceptance 函数。
        """

        acceptance_fn: Optional[Callable[[int], float]] = None
        if acceptance_rate is not None:
            acceptance_fn = lambda _i: acceptance_rate

        result: SimulationResult = self.simulator.simulate(
            total_batch_size=batch_size,
            micro_batch_size=batch_size,
            k_steps=k_steps,
            partition_point=0,
            prompt_len=prompt_len,
            acceptance_fn=acceptance_fn,
        )

        self.current_seq_len += result.effective_tokens_per_seq
        self.generated_tokens += result.effective_tokens_per_seq

        return result.makespan, result.effective_tokens_per_seq

    def run_benchmark(
        self,
        total_tokens: int,
        k_steps: int,
        alpha: Optional[float] = None,
        batch_size: int = 1,
        prompt_len: int = 512,
    ):
        """
        持续运行直到生成足够 token。
        """
        elapsed_time = 0.0
        tokens_produced = 0.0

        while tokens_produced < total_tokens:
            latency, new_tokens = self.simulate_step(
                k_steps=k_steps,
                acceptance_rate=alpha,
                batch_size=batch_size,
                prompt_len=prompt_len,
            )
            elapsed_time += latency
            tokens_produced += new_tokens

        throughput = tokens_produced / elapsed_time if elapsed_time > 0 else 0.0
        return elapsed_time, throughput