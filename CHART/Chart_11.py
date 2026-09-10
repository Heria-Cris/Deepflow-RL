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
# 2. 数据清洗与准备
# ==========================================
# 提取变化的核心变量，将固定的 P=0, K=1 移至全局说明中
actions = ["MB = 1", "MB = 2", "MB = 4", "MB = 8"]
counts = [16, 4, 8, 8]
total_cells = sum(counts)  # 36

# ==========================================
# 3. 跨图色彩锁定 (与 Fig 10 热力图保持绝对一致)
# ==========================================
# 1=翡翠绿, 2=深紫, 4=学术蓝, 8=深红
colors = ['#16a085', '#8e44ad', '#2980b9', '#c0392b']

fig, ax = plt.subplots(figsize=(8.5, 5))

# ==========================================
# 4. 绘制精致的黑框柱状图
# ==========================================
width = 0.55
bars = ax.bar(actions, counts, color=colors, edgecolor='black',
              linewidth=1.2, width=width, zorder=3)

# ==========================================
# 5. 图表修饰 (Data-Ink Ratio 优化)
# ==========================================
ax.set_ylabel("Selection Count (out of 36)", fontsize=13, fontweight='bold', labelpad=10)
# 将 P=0, K=1 作为 X 轴的副标题，大幅净化刻度标签
ax.set_xlabel("Action (Micro-Batch Size) under fixed P=0, K=1", fontsize=13, fontweight='bold', labelpad=10)

ax.tick_params(axis='both', labelsize=11.5)

# 移除顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 将网格线置于柱子后方
ax.grid(axis="y", linestyle="--", alpha=0.5, color='#bdc3c7', zorder=0)
ax.set_axisbelow(True)

# 留出顶部空间打标签
ax.set_ylim(0, max(counts) * 1.3)

# ==========================================
# 6. 点睛之笔：添加带百分比的悬浮数据标签
# ==========================================
for bar, v in zip(bars, counts):
    pct = (v / total_cells) * 100
    # 在数值下方追加百分比，这是顶会论文中最受欢迎的数据展示方式
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
            f"{v}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=11, fontweight='bold', color='black')

# ==========================================
# 7. 讲故事 (Storytelling Callout)
# ==========================================
# 解释为什么 MB=1 占比最高：因为在大量的组合中，系统倾向于快速启动流水线
ax.annotate("Dominant Action\n(Prefill/Compute Bound)",
            xy=(bars[0].get_x() + bars[0].get_width()/2 + 0.1, counts[0]),
            xytext=(bars[0].get_x() + bars[0].get_width()/2 + 1.2, counts[0] - 1),
            arrowprops=dict(facecolor='#16a085', edgecolor='#16a085', arrowstyle='->', lw=1.5),
            fontsize=11, style='italic', color='#16a085', ha='center')

ax.set_title("Distribution of PPO Agent Actions in Policy Grid",
             fontsize=14, fontweight='bold', pad=25)

# ==========================================
# 8. 保存与输出
# ==========================================
plt.tight_layout()
plt.savefig("fig11_unique_action_distribution.pdf", bbox_inches="tight")
plt.savefig("fig11_unique_action_distribution.png", dpi=600, bbox_inches="tight")
plt.show()