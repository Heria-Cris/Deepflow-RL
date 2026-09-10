# test_phase2.py
import json

from core.hardware import Device, NetworkLink
from core.model_spec import LLaMAModel
from engine.speculative import SpeculativeScheduler


def run_phase2_experiment():
    print("=" * 60)
    print("🧪 DeepFlow-RL - Phase 2: Unified K-Sweep")
    print("=" * 60)

    with open("configs/devices_paper.json", "r") as f:
        dev_cfg = json.load(f)

    edge = Device(**dev_cfg["devices"][0])
    cloud = Device(**dev_cfg["devices"][1])
    network = NetworkLink(**dev_cfg["network"])

    target_model = LLaMAModel("configs/llama2_7b_paper.json")
    draft_model = LLaMAModel("configs/llama_1b_paper.json")

    scheduler = SpeculativeScheduler(
        draft_model=draft_model,
        target_model=target_model,
        edge_device=edge,
        cloud_device=cloud,
        network=network,
    )

    k_values = [0, 1, 3, 5, 7, 10]
    total_gen_tokens = 50
    prompt_len = 512

    results = []

    print(f"[Config] Network: {network.bandwidth_mbps} Mbps, RTT: {network.latency_s * 1000:.1f} ms")
    print(f"[Config] Prompt Length: {prompt_len}")
    print("-" * 75)
    print(f"{'K':<8} | {'Latency(s)':<15} | {'Throughput(tok/s)':<20} | {'Speedup':<10}")
    print("-" * 75)

    baseline_tps = None

    for k in k_values:
        elapsed_time, tps = scheduler.run_benchmark(
            total_tokens=total_gen_tokens,
            k_steps=k,
            alpha=None,          # 使用统一 acceptance 函数
            batch_size=1,
            prompt_len=prompt_len,
        )

        if baseline_tps is None:
            baseline_tps = tps

        speedup = tps / baseline_tps if baseline_tps and baseline_tps > 0 else 1.0
        results.append((k, elapsed_time, tps, speedup))
        print(f"{k:<8} | {elapsed_time:<15.4f} | {tps:<20.4f} | {speedup:<10.2f}x")

    best_k = max(results, key=lambda x: x[2])[0]
    print("-" * 75)
    print(f"✅ Best K under this unified backend: K={best_k}")


if __name__ == "__main__":
    run_phase2_experiment()