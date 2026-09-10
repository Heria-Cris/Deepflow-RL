# test_phase3.py
import json

from core.hardware import Device, NetworkLink
from core.model_spec import LLaMAModel
from engine.pipeline import PipelineScheduler


def run_phase3_experiment():
    print("=" * 60)
    print("🏭 DeepFlow-RL - Phase 3: Unified Pipeline Experiment")
    print("=" * 60)

    with open("configs/devices_paper.json", "r") as f:
        dev_cfg = json.load(f)

    edge = Device(**dev_cfg["devices"][0])
    cloud = Device(**dev_cfg["devices"][1])
    network = NetworkLink(**dev_cfg["network"])

    target_model = LLaMAModel("configs/llama2_7b_paper.json")
    draft_model = LLaMAModel("configs/llama_1b_paper.json")

    scheduler = PipelineScheduler(
        draft_model=draft_model,
        target_model=target_model,
        edge=edge,
        cloud=cloud,
        network=network,
    )

    total_batch = 32
    k_steps = 5
    prompt_len = 512

    print(f"Task: batch={total_batch}, K={k_steps}, prompt_len={prompt_len}")
    print("-" * 60)

    mb_seq = 32
    time_seq, _, metrics_seq = scheduler.schedule(
        total_batch_size=total_batch,
        micro_batch_size=mb_seq,
        k_steps=k_steps,
        prompt_len=prompt_len,
        partition_point=0,
    )

    print(f"[Scenario A] Sequential (Micro-batch={mb_seq})")
    print(f"  - Makespan   : {time_seq:.4f} s")
    print(f"  - Throughput : {metrics_seq['throughput']:.2f} tok/s")
    print(f"  - Cloud Util : {metrics_seq['util_cloud'] * 100:.1f}%")

    mb_pipe = 4
    time_pipe, timeline, metrics_pipe = scheduler.schedule(
        total_batch_size=total_batch,
        micro_batch_size=mb_pipe,
        k_steps=k_steps,
        prompt_len=prompt_len,
        partition_point=0,
    )

    print(f"\n[Scenario B] Pipelined (Micro-batch={mb_pipe})")
    print(f"  - Makespan   : {time_pipe:.4f} s")
    print(f"  - Throughput : {metrics_pipe['throughput']:.2f} tok/s")
    print(f"  - Cloud Util : {metrics_pipe['util_cloud'] * 100:.1f}%")

    speedup = metrics_pipe["throughput"] / metrics_seq["throughput"] if metrics_seq["throughput"] > 0 else 0.0
    print("-" * 60)
    print(f"🚀 Pipeline Speedup: {speedup:.2f}x")

    scheduler.print_timeline(timeline)


if __name__ == "__main__":
    run_phase3_experiment()