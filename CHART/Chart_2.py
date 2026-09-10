import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 全局学术风设置 (字体与线宽)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# ==========================================
# 2. 载入新实验数据 (Phase 2: Unified K-Sweep)
# ==========================================
K = [0, 1, 3, 5, 7, 10]
latency = [4.6171, 3.7808, 3.5317, 3.9879, 4.3244, 5.2331]
throughput = [10.8293, 13.7006, 14.4367, 13.5359, 12.3370, 10.6076]
speedup = [1.00, 1.27, 1.33, 1.25, 1.14, 0.98]

# ==========================================
# 3. 高级莫兰迪/经典学术配色
# ==========================================
color_lat = '#c0392b'  # 深砖红 (代表延迟/开销)
color_thr = '#2980b9'  # 海军蓝 (代表吞吐量/收益)

fig, ax1 = plt.subplots(figsize=(8.5, 5.5))
ax2 = ax1.twinx()

# ==========================================
# 4. 绘制折线图
# ==========================================
l1 = ax1.plot(
    K, latency,
    marker="o", markersize=8.5, mfc='white', mew=1.6,
    color=color_lat, linewidth=2.2, label="Latency (s)"
)

l2 = ax2.plot(
    K, throughput,
    marker="s", markersize=8.5, mfc='white', mew=1.6,
    color=color_thr, linewidth=2.2, label="Throughput (tok/s)"
)

# ==========================================
# 5. 坐标轴与刻度精调
# ==========================================
ax1.set_xlabel("Speculative Depth ($K$)", fontsize=17, fontweight='bold', labelpad=10)
ax1.set_ylabel("Latency (seconds)", fontsize=17, fontweight='bold', color=color_lat)
ax2.set_ylabel("Throughput (tokens/s)", fontsize=17, fontweight='bold', color=color_thr)

ax1.set_xticks(K)
ax1.set_xticklabels(K, fontsize=15)
ax1.tick_params(axis='y', labelsize=16, colors=color_lat)
ax2.tick_params(axis='y', labelsize=16, colors=color_thr)
ax1.tick_params(axis='x', labelsize=16)

ax1.set_ylim(3.0, 5.8)
ax2.set_ylim(9.0, 16.0)

# ==========================================
# 6. 图表修饰
# ==========================================
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

ax1.grid(axis='y', linestyle="--", alpha=0.4, color='#7f8c8d')
ax1.set_axisbelow(True)

# 保留图例，但删除图注说明和 sweet spot 标记
lines = l1 + l2
labels = [l.get_label() for l in lines]
ax1.legend(
    lines, labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.14),
    ncol=2,
    frameon=False,
    fontsize=16
)

# ==========================================
# 7. Speedup 文本标签
# ==========================================
for k_val, thr_val, spd_val in zip(K, throughput, speedup):
    ax2.annotate(
        f"{spd_val:.2f}x",
        (k_val, thr_val),
        textcoords="offset points",
        xytext=(0, 12),
        ha='center',
        fontsize=15,
        color=color_thr,
        fontweight='bold'
    )

# ==========================================
# 8. 保存与输出
# ==========================================
plt.tight_layout()
plt.savefig("fig2_k_sweep.pdf", bbox_inches="tight")
plt.savefig("fig2_k_sweep.png", dpi=600, bbox_inches="tight")
plt.show()