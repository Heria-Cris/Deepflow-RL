import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# ==========================================
# 1. 全局学术风设置
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# ==========================================
# 2. 数据重组与逻辑分组 (极其重要)
# ==========================================
# 按 Prompt 长度进行逻辑分组，体现 "Shift" 的过程
# 组 1: Short Prompts (全是 Edge)
group1_labels = ["Weak Net\n(Short)", "Moderate Net\n(Short)", "Strong Net\n(Short)", "High RTT\n(Short)"]
group1_vals = [0, 0, 0, 0]

# 组 2: Long Prompts (全是 Cloud)
group2_labels = ["Weak Net\n(Long)", "Strong Net\n(Long)"]
group2_vals = [1, 1]  # 将 2 改为 1，更紧凑

# 组 3: Very Long Prompts (全是 Cloud)
group3_labels = ["Weak Net\n(VLong)", "Strong Net\n(VLong)", "High RTT\n(VLong)"]
group3_vals = [1, 1, 1]

all_labels = group1_labels + group2_labels + group3_labels
all_vals = group1_vals + group2_vals + group3_vals

x = np.arange(len(all_labels))

# ==========================================
# 3. 开始绘图 (使用离散状态散点/连线图)
# ==========================================
fig, ax = plt.subplots(figsize=(10.5, 5))

# 配色：边缘(Edge)为治愈绿，云端(Cloud)为深邃蓝
color_edge = '#27ae60'
color_cloud = '#2980b9'
colors = [color_edge if v == 0 else color_cloud for v in all_vals]

# 绘制带有黑框的高级大空心散点 (LaTeX pgfplots 质感)
scatter = ax.scatter(x, all_vals, c=colors, s=300, edgecolor='black',
                     linewidth=1.5, zorder=4)

# 添加一条柔和的灰色连线，引导读者的视线完成 "转移 (Shift)" 的轨迹
ax.plot(x, all_vals, color='#bdc3c7', linestyle='--', linewidth=2, zorder=2)

# ==========================================
# 4. 图表修饰 (Data-Ink Ratio 优化)
# ==========================================
ax.set_yticks([0, 1])
# 加粗加大的 Y 轴标签，彻底取代冗余的数据点文本
ax.set_yticklabels(["Edge Device\n(Compute Bound)", "Cloud Server\n(Memory/Compute Bound)"],
                   fontsize=12, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(all_labels, fontsize=10.5, rotation=0)

ax.set_ylim(-0.5, 1.5)  # 留出上下空间，避免点贴边
ax.set_xlim(-0.5, len(all_labels) - 0.5)

# 移除三面边框，仅保留底部
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# 将 Y 轴刻度短线隐藏，更加清爽
ax.tick_params(axis='y', which='both', length=0)
ax.grid(axis='y', linestyle='-', alpha=0.3, color='#95a5a6', zorder=1)

# ==========================================
# 5. 点睛之笔：添加逻辑分组的高亮底纹 (Visual Grouping)
# ==========================================
# 背景分块色带：突出 Short -> Long 的质变
# Short 区域 (x= -0.5 到 3.5)
ax.axvspan(-0.5, 3.5, facecolor='#27ae60', alpha=0.05, zorder=0)
ax.text(1.5, 1.3, "Phase 1: Short Prompts\n(Edge Bottleneck)",
        ha='center', va='center', fontsize=12, fontweight='bold', color=color_edge)

# Long & VLong 区域 (x= 3.5 到 8.5)
ax.axvspan(3.5, 8.5, facecolor='#2980b9', alpha=0.05, zorder=0)
ax.text(6, 1.3, "Phase 2: Long & Very Long Prompts\n(Cloud Bottleneck Shift)",
        ha='center', va='center', fontsize=12, fontweight='bold', color=color_cloud)

# 绘制一条垂直虚线作为分界点 (Phase Transition)
ax.axvline(x=3.5, color='#7f8c8d', linestyle=':', linewidth=2, zorder=3)
ax.annotate("Bottleneck\nShift Point", xy=(3.5, 0.5), xytext=(2.5, 0.5),
            arrowprops=dict(facecolor='#7f8c8d', edgecolor='#7f8c8d', arrowstyle='->'),
            color='#7f8c8d', fontsize=11, fontweight='bold', ha='center', va='center')

# ==========================================
# 6. 保存与输出
# ==========================================
ax.set_title("System Bottleneck Shift across Varying Prompt Lengths",
             fontsize=14, fontweight='bold', pad=25)

plt.tight_layout()
plt.savefig("fig13_bottleneck_shift.pdf", bbox_inches="tight")
plt.savefig("fig13_bottleneck_shift.png", dpi=600, bbox_inches="tight")
plt.show()