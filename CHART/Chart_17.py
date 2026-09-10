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
# 2. 数据准备
# ==========================================
prompt = ["128", "512", "1024", "2048", "4096"]
edge_oom = np.array([0, 36, 144, 288, 366])
cloud_oom = np.array([0, 0, 0, 41, 48])
both_oom = np.array([0, 0, 0, 13, 126])

totals = edge_oom + cloud_oom + both_oom
x = np.arange(len(prompt))

# ==========================================
# 3. 跨图语义锁色 (Semantic Color Locking)
# ==========================================
# 保持与 Fig 13 (Bottleneck Shift) 的物理语境绝对一致！
color_edge = '#27ae60'  # 边缘端: 治愈绿
color_cloud = '#2980b9' # 云端: 深邃蓝
color_both = '#c0392b'  # 双端崩溃: 灾难红 (极具视觉冲击力)

fig, ax = plt.subplots(figsize=(8.5, 5.5))

# ==========================================
# 4. 绘制高级堆叠柱状图 (带黑色描边)
# ==========================================
width = 0.55
# 绘制边缘 OOM (底层)
b1 = ax.bar(x, edge_oom, width, color=color_edge, edgecolor='black',
            linewidth=1.2, label="Edge OOM", zorder=3)
# 绘制云端 OOM (中层)
b2 = ax.bar(x, cloud_oom, width, bottom=edge_oom, color=color_cloud,
            edgecolor='black', linewidth=1.2, label="Cloud OOM", zorder=3)
# 绘制双端 OOM (顶层)
b3 = ax.bar(x, both_oom, width, bottom=edge_oom + cloud_oom, color=color_both,
            edgecolor='black', linewidth=1.2, label="Both OOM (Critical)", zorder=3)

# ==========================================
# 5. 图表修饰与留白 (Data-Ink Ratio)
# ==========================================
ax.set_xticks(x)
ax.set_xticklabels(prompt, fontsize=17, fontweight='bold')
ax.tick_params(axis='y', labelsize=17)

ax.set_xlabel("Prompt Length", fontsize=17, fontweight='bold', labelpad=10)
ax.set_ylabel("Number of Infeasible Actions (OOM)", fontsize=17, fontweight='bold', labelpad=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.grid(axis="y", linestyle="--", alpha=0.5, color='#bdc3c7', zorder=0)
ax.set_axisbelow(True)

# 动态计算完美 Y 轴高度 (给顶部 Total 标签和说明文字留足 20% 空间)
ax.set_ylim(0, max(totals) * 1.25)

# ==========================================
# 6. 点睛之笔：智能文本打标系统 (Smart Labeling)
# ==========================================
for i in range(len(x)):
    # A. 柱顶总数悬浮标签 (Total Anchors)
    if totals[i] > 0:
        ax.text(x[i], totals[i] + 12, f"Total: {totals[i]}",
                ha='center', va='bottom', fontsize=13, fontweight='bold', color='black')
    elif totals[i] == 0:
        ax.text(x[i], 12, "0 OOM\n(All Feasible)",
                ha='center', va='bottom',
                fontsize=12,
                fontweight='bold',
                style='italic',
                color='#f39c12')

    # B. 段内高对比打标 (Inner Segment Labels)
    # 只在区块高度足够时 (例如 > 20) 才打字，避免小区块文字溢出重叠
    if edge_oom[i] > 20:
        ax.text(x[i], edge_oom[i] / 2, str(edge_oom[i]),
                ha='center', va='center', fontsize=13, color='white', fontweight='bold')
    if cloud_oom[i] > 20:
        ax.text(x[i], edge_oom[i] + cloud_oom[i] / 2, str(cloud_oom[i]),
                ha='center', va='center', fontsize=13, color='white', fontweight='bold')
    if both_oom[i] > 20:
        ax.text(x[i], edge_oom[i] + cloud_oom[i] + both_oom[i] / 2, str(both_oom[i]),
                ha='center', va='center', fontsize=13, color='white', fontweight='bold')

# ==========================================
# 8. 图例与保存
# ==========================================
# 图例水平置于顶部，去除边框
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=3, frameon=False, fontsize=15)


plt.tight_layout()
plt.savefig("fig17_oom_distribution.pdf", bbox_inches="tight")
plt.savefig("fig17_oom_distribution.png", dpi=600, bbox_inches="tight")
plt.show()