# test_phase1.py
import json
import os

from core.hardware import Device, NetworkLink
from core.model_spec import LLaMAModel


def run_academic_test():
    print("=" * 60)
    print("🔬 DeepFlow-RL v3.0 - Phase 1 Academic Verification")
    print("=" * 60)

    # 路径检查
    if not os.path.exists("configs/llama2_7b_paper.json"):
        print("Error: Config files not found. Please create 'configs/llama2_7b_paper.json'.")
        return

    # 论文主场景统一口径：Weak Network = 1 Mbps / 50 ms
    MAIN_WEAK_BW_MBPS = 1.0
    MAIN_WEAK_LAT_MS = 50.0

    # 1. 初始化硬件集群
    print("\n[Step 1] Loading Hardware Cluster...")
    with open("configs/devices_paper.json", "r", encoding="utf-8") as f:
        dev_cfg = json.load(f)

    devices = {}
    for d_params in dev_cfg["devices"]:
        dev = Device(**d_params)
        devices[dev.device_type] = dev
        print(f"  -> Loaded: {dev}")

    # 显式覆盖为论文主弱网场景，避免和默认配置口径冲突
    network = NetworkLink(
        bandwidth_mbps=MAIN_WEAK_BW_MBPS,
        latency_ms=MAIN_WEAK_LAT_MS,
        bandwidth_efficiency=dev_cfg["network"].get("bandwidth_efficiency", 0.80),
        serialization_overhead_ms=dev_cfg["network"].get("serialization_overhead_ms", 0.0),
    )
    print(
        f"  -> Network (Main Weak-Net Scenario): "
        f"{network.bandwidth_mbps:.1f} Mbps, Latency: {network.latency_s * 1000:.1f} ms"
    )

    # 2. 加载模型
    print("\n[Step 2] Loading Analytical Model Cost Model...")
    target_model = LLaMAModel("configs/llama2_7b_paper.json")
    print(f"  -> Model: {target_model.name} ({len(target_model.layers)} layers)")

    # 3. 核心实验：Prefill vs Decode 的物理差异
    print("\n[Step 3] Simulation: Prefill vs Decode Physics")

    batch_size = 1
    seq_len = 512
    layer = target_model.layers[5]
    edge_dev = devices["edge"]

    # ------------------------------------------------------------------
    # Scenario A: Prefill
    # ------------------------------------------------------------------
    flops = layer.get_flops(batch_size, seq_len, is_decoding=False)
    compute_time = edge_dev.compute_time(flops)
    act_size = layer.get_activation_memory_mb(batch_size, seq_len)
    comm_time = network.estimate_comm_time(act_size)

    print(f"\nScenario A: Prefill (Seq={seq_len}) on Edge")
    print(f"  - Compute Time : {compute_time * 1000:.2f} ms (FLOPs: {flops:.4f} G)")
    print(f"  - Comm Time    : {comm_time:.2f} s (Size: {act_size:.2f} MB)")
    print(
        "  -> 结论: 在主弱网场景（1 Mbps / 50 ms）下，"
        "Prefill 激活值跨设备传输代价极高，因此更适合把 Prefill 留在本地执行，"
        "或进一步压缩中间状态。"
    )

    # ------------------------------------------------------------------
    # Scenario B: Decode + Speculative
    # ------------------------------------------------------------------
    spec_k = 5

    # Decode 端计算量很小，通常更接近 memory/IO bound
    decode_flops = layer.get_flops(batch_size, seq_len, is_decoding=True) * spec_k
    _ = decode_flops  # 保留变量，便于后续扩展分析

    weight_size_mb = layer.get_parameter_count() * 2 / 1024 ** 2
    io_time = edge_dev.memory_access_time(weight_size_mb) * spec_k

    # 传统方案：传激活值
    act_size_decode = layer.get_activation_memory_mb(batch_size, spec_k)
    comm_time_trad = network.estimate_comm_time(act_size_decode)

    # DeepFlow：传 token IDs
    token_id_size = target_model.get_token_id_size_mb(batch_size, spec_k)
    comm_time_spec = network.estimate_comm_time(token_id_size)

    print(f"\nScenario B: Speculative Decode (K={spec_k})")
    print(f"  - Local IO Time: {io_time * 1000:.2f} ms (受显存带宽限制)")
    print(f"  - [Legacy] Transmit Activations: {comm_time_trad:.4f} s")
    print(f"  - [Ours]   Transmit Token IDs  : {comm_time_spec:.4f} s")

    speedup = comm_time_trad / max(comm_time_spec, 1e-12)
    print(
        f"  -> 🚀 在主弱网场景（1 Mbps / 50 ms）下，"
        f"使用 Token-ID 传输相对传统激活值传输可带来约 {speedup:.1f}x 的通信侧收益。"
    )


if __name__ == "__main__":
    run_academic_test()