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
# 2. 数据与配色
# ==========================================
methods = [
    "Local Only",
    "Cloud Only",
    "Legacy Split",
    "Static DeepFlow",
    "PPO DeepFlow"
]
throughput = [3.59, 23.05, 10.55, 39.53, 39.53]

# 每个柱子单独一种图例颜色
colors = ['#5f6b6d', '#8f9a9c', '#c7cdcf', '#2980b9', '#c0392b']

x = np.arange(len(methods))

fig, ax = plt.subplots(figsize=(9.5, 5.5))

# ==========================================
# 3. 绘制柱状图
# ==========================================
bars = []
for i in range(len(methods)):
    bar = ax.bar(
        x[i], throughput[i],
        color=colors[i],
        edgecolor='black',
        linewidth=1.2,
        width=0.55,
        zorder=3,
        label=methods[i]
    )
    bars.append(bar[0])

# ==========================================
# 4. 图表修饰
# ==========================================
ax.set_ylabel("Throughput (tok/s)", fontsize=17, fontweight='bold', labelpad=10)
ax.tick_params(axis='y', labelsize=17)

# 删除 x 轴文字说明
ax.set_xticks(x)
ax.set_xticklabels([])
ax.tick_params(axis='x', length=0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.grid(axis="y", linestyle="--", alpha=0.5, color='#bdc3c7', zorder=0)
ax.set_axisbelow(True)

ax.set_ylim(0, 52)

# ==========================================
# 6. 数据标签与标注
# ==========================================
for i, (bar, v) in enumerate(zip(bars, throughput)):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        v + 0.8,
        f"{v:.2f}",
        ha="center",
        va="bottom",
        fontsize=15,
        fontweight='bold',
        color='black'
    )

# ==========================================
# 7. 图例：每个柱子单独说明
# ==========================================
ax.legend(
    handles=bars,
    labels=methods,
    loc='upper left',
    frameon=False,
    fontsize=16
)

# ==========================================
# 8. 保存与输出
# ==========================================
plt.tight_layout()
plt.savefig("fig6_overall_comparison.pdf", bbox_inches="tight")
plt.savefig("fig6_overall_comparison.png", dpi=600, bbox_inches="tight")
plt.show()