import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 全局学术风设置
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# ==========================================
# 2. 数据与配色准备
# ==========================================
scenarios = ["Weak Net", "Strong Net"]
legacy_tp = np.array([4.24, 6.28])
deepflow_tp = np.array([14.09, 14.15])

legacy_dt = np.array([0.2500, 0.2500])
deepflow_dt = np.array([0.0009, 0.0009])

# 配色：Legacy为基线(灰)，DeepFlow为提出方案(深蓝)
color_legacy = '#95a5a6'
color_deepflow = '#2980b9'

x = np.arange(len(scenarios))
w = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5))  # 略微加宽以容纳两图

# ==========================================
# 3. 绘制左图：吞吐量 (Throughput)
# ==========================================
b1 = ax1.bar(x - w / 2, legacy_tp, width=w, label="Legacy Split",
             color=color_legacy, edgecolor='black', linewidth=1.2, zorder=3)
b2 = ax1.bar(x + w / 2, deepflow_tp, width=w, label="DeepFlow Expert",
             color=color_deepflow, edgecolor='black', linewidth=1.2, zorder=3)

ax1.set_xticks(x)
ax1.set_xticklabels(scenarios, fontsize=17, fontweight='bold')
ax1.set_ylabel("Throughput (tok/s)", fontsize=17, fontweight='bold')
ax1.set_title("(a) Throughput Comparison", fontsize=17, fontweight='bold', pad=15)
ax1.set_ylim(0, max(deepflow_tp) * 1.3)  # 留出顶部空间打标签

# 左图打标签与加速比 (Speedup)
for i in range(len(scenarios)):
    # 数值标签
    ax1.text(x[i] - w / 2, legacy_tp[i] + 0.3, f"{legacy_tp[i]:.2f}", ha='center', va='bottom', fontsize=15)
    ax1.text(x[i] + w / 2, deepflow_tp[i] + 0.3, f"{deepflow_tp[i]:.2f}", ha='center', va='bottom', fontsize=15,
             fontweight='bold', color=color_deepflow)

# ==========================================
# 4. 绘制右图：数据传输量 (Data Transfer)
# ==========================================
b3 = ax2.bar(x - w / 2, legacy_dt, width=w, color=color_legacy, edgecolor='black', linewidth=1.2, zorder=3)
b4 = ax2.bar(x + w / 2, deepflow_dt, width=w, color=color_deepflow, edgecolor='black', linewidth=1.2, zorder=3)

# 核心修改：设置对数坐标轴 (Log Scale)，拯救被湮灭的 0.0009
ax2.set_yscale("log")
ax2.set_ylim(0.0003, 2.0)  # 仔细调整上下限，使得两根柱子都好看

ax2.set_xticks(x)
ax2.set_xticklabels(scenarios, fontsize=11, fontweight='bold')
ax2.set_ylabel("Data Transfer (MB, Log Scale)", fontsize=17, fontweight='bold')
ax2.set_title("(b) Transferred Data Comparison", fontsize=17, fontweight='bold', pad=15)

# 右图打标签与减少倍数 (Reduction)
for i in range(len(scenarios)):
    # 数值标签 (对数轴上的位置需要用乘法)
    ax2.text(x[i] - w / 2, legacy_dt[i] * 1.15, f"{legacy_dt[i]:.4f}", ha='center', va='bottom', fontsize=15)
    ax2.text(x[i] + w / 2, deepflow_dt[i] * 1.15, f"{deepflow_dt[i]:.4f}", ha='center', va='bottom', fontsize=15,
             fontweight='bold', color=color_deepflow)

# ==========================================
# 5. 全局修饰与图例 (Data-Ink Ratio)
# ==========================================
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color='#bdc3c7', zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', labelsize=17)
    ax.tick_params(axis='x', labelsize=17)

# 提取全局图例，放在整张图片的正上方居中位置
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.07),
           ncol=2, frameon=False, fontsize=17)

# ==========================================
# 6. 保存与输出
# ==========================================
plt.tight_layout()
plt.savefig("fig5_legacy_vs_deepflow.pdf", bbox_inches="tight")
plt.savefig("fig5_legacy_vs_deepflow.png", dpi=600, bbox_inches="tight")
plt.show()