import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ==========================================
# 1. 全局学术风设置
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# Data
labels = [
    "Prefill\nCompute",
    "Prefill\nActivation Comm",
    "Decode\nActivation Comm",
    "Decode\nToken-ID Comm",
]
values = [0.00856, 39.07, 0.4316, 0.0507]  # seconds

# ==========================================
# 2. 配色
# ==========================================
colors = ['#95a5a6', '#c0392b', '#e74c3c', '#2980b9']

# 画布稍微加宽一点，适合论文双栏缩放
fig, ax = plt.subplots(figsize=(8.8, 5.8))

# 绘制柱状图
bars = ax.bar(
    labels, values,
    color=colors,
    edgecolor='black',
    linewidth=1.2,
    width=0.52,
    zorder=3
)

# ==========================================
# 3. 对数坐标轴与网格
# ==========================================
ax.set_yscale("log")
ax.set_ylim(0.003, 400)   # 顶部留更多空间，避免最高柱标签太挤

formatter = ticker.FuncFormatter(lambda y, _: f"{y:g}")
ax.yaxis.set_major_formatter(formatter)

ax.grid(axis='y', which='major', linestyle='-', alpha=0.35, color='#7f8c8d', zorder=0)
ax.grid(axis='y', which='minor', linestyle=':', alpha=0.18, color='#7f8c8d', zorder=0)
ax.set_axisbelow(True)

# ==========================================
# 4. 图表修饰
# ==========================================
ax.set_ylabel("Time (seconds, Log Scale)", fontsize=17, fontweight='bold', labelpad=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 刻度字体略放大，但避免过大拥挤
ax.tick_params(axis='x', labelsize=14, pad=6)
ax.tick_params(axis='y', labelsize=15)

# ==========================================
# 5. 数据标签（重点放大）
# ==========================================
for bar, v in zip(bars, values):
    if v < 1:
        text_label = f"{v:.4f}s"
    else:
        text_label = f"{v:.2f}s"

    # 对数坐标下，不同高度使用不同倍数偏移，避免贴柱子或贴天花板
    if v < 0.02:
        y_text = v * 1.28
    elif v < 1:
        y_text = v * 1.22
    else:
        y_text = v * 1.18

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        y_text,
        text_label,
        ha="center",
        va="bottom",
        fontsize=16,
        fontweight='bold',
        color='black'
    )

# ==========================================
# 6. 边距与保存
# ==========================================
plt.tight_layout(pad=1.0)
plt.savefig("fig1_activation_vs_tokenid_cost.pdf", bbox_inches="tight")
plt.savefig("fig1_activation_vs_tokenid_cost.png", dpi=600, bbox_inches="tight")
plt.show()