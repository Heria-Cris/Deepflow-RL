import numpy as np
import matplotlib.pyplot as plt

# --- 终极参数配置 ---

# 1. 模拟弱边缘设备 (关键！)
# 将算力降到 2.0 TFLOPS (约等于 Jetson Nano 或高端手机 NPU)
EDGE_TFLOPS = 2.0

# 2. 接受率衰减 (让 K=10 的收益没那么大)
ALPHA_MAP = {
    0: 0.0,
    1: 0.95,
    3: 0.85,
    5: 0.70,
    7: 0.55,
    10: 0.40  # 稍微降一点，给 K=7 机会
}

# 固定参数
BATCH_SIZE = 32
CLOUD_TFLOPS = 312.0
# 模型参数 (TinyLlama 1.1B)
DRAFT_FLOPS_PER_TOKEN = 2.2  # GFLOPs approx
TARGET_FLOPS_PER_TOKEN = 14.0  # LLaMA 7B


def calculate_throughput(rtt_ms, k):
    # 1. Edge Compute Time (Serial)
    # Time = FLOPs / (TFLOPS * 1000)
    t_draft = (k * DRAFT_FLOPS_PER_TOKEN) / (EDGE_TFLOPS * 1000) if k > 0 else 0

    # 2. Network Time
    # Token ID size is negligible, mainly RTT
    t_comm = rtt_ms / 1000.0

    # 3. Cloud Verify Time (Parallel)
    # Process K+1 tokens
    t_verify = ((k + 1) * TARGET_FLOPS_PER_TOKEN) / (CLOUD_TFLOPS * 1000)

    # Total Latency
    makespan = t_draft + t_comm + t_verify

    # Effective Tokens
    alpha = ALPHA_MAP.get(k, 0.3)
    effective_tokens = 1 + k * alpha

    # System Throughput
    return (BATCH_SIZE * effective_tokens) / makespan


def run_diagnosis():
    latencies = [5, 10, 20, 50, 100, 200]
    k_options = [0, 1, 3, 5, 7, 10]

    best_ks = []

    print(f"{'RTT(ms)':<10} | {'Best K':<10} | {'Throughput Details'}")
    print("-" * 60)

    for lat in latencies:
        scores = []
        for k in k_options:
            tps = calculate_throughput(lat, k)
            scores.append(tps)

        # 找到吞吐量最高的 K
        best_k = k_options[np.argmax(scores)]
        best_ks.append(best_k)

        # 打印详细得分以便分析
        details = ", ".join([f"K{k}:{int(s)}" for k, s in zip(k_options, scores)])
        print(f"{lat:<10} | {best_k:<10} | {details}")

    # 画图
    plt.figure(figsize=(8, 5))
    plt.plot(latencies, best_ks, marker='o', linestyle='-', color='b')
    plt.xlabel("Network Latency RTT (ms)")
    plt.ylabel("Optimal Speculative Steps (K)")
    plt.title(f"Optimal K vs Latency (Edge TFLOPS={EDGE_TFLOPS})")
    plt.grid(True)
    plt.yticks(k_options)
    plt.show()


if __name__ == "__main__":
    run_diagnosis()