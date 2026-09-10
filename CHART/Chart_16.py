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
prompt = np.array([128, 512, 1024, 2048, 4096])
feasible_ratio = np.array([100.00, 96.97, 87.88, 71.21, 54.55])

fig, ax = plt.subplots(figsize=(8.5, 5.5))

# ==========================================
# 3. 高级配色系统与阴影填充 (绝杀技)
# ==========================================
color_line = '#2980b9'  # 学术蓝 (代表安全/可行的空间)
color_pruned = '#e74c3c' # 警示红 (代表被裁剪的 OOM 空间)

# 技巧 A：填充 100% 与折线之间的区域，代表“被 Mask 掉的无效动作”
ax.fill_between(prompt, feasible_ratio, 100, color=color_pruned, alpha=0.15,
                hatch='//', edgecolor=color_pruned, label="Pruned OOM Actions")

# 技巧 B：填充折线以下的区域，代表“安全的动作空间”
ax.fill_between(prompt, 0, feasible_ratio, color=color_line, alpha=0.1,
                label="Feasible Action Space")

# 绘制带有高级空心 Marker 的主体折线
ax.plot(prompt, feasible_ratio, marker="o", markersize=9, mfc='white', mew=1.5,
        color=color_line, linewidth=2.5, zorder=3)

# ==========================================
# 4. 核心修复：X轴对数化 (Log-2 Scale)
# ==========================================
# Prompt 是按 2 的倍数甚至 4 倍数跳跃的，必须用 Log 轴才能展示正确的衰减斜率
ax.set_xscale('log', base=2)
ax.set_xticks(prompt)
ax.set_xticklabels([str(p) for p in prompt], fontsize=17)
ax.tick_params(axis='y', labelsize=17)

# ==========================================
# 5. 图表修饰 (Data-Ink Ratio 优化)
# ==========================================
ax.set_xlabel("Prompt Length (Log Scale)", fontsize=17, fontweight='bold', labelpad=10)
ax.set_ylabel("Feasible Action Ratio (%)", fontsize=17, fontweight='bold', labelpad=10)

# 设置完美的 Y 轴视口范围 (突出收缩感)
ax.set_ylim(45, 105)
ax.set_xlim(115, 4500) # 给最左侧和最右侧留一点点边距

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 次级网格线增强对数轴质感
ax.grid(axis='y', linestyle="--", alpha=0.4, color='#95a5a6', zorder=0)
ax.grid(axis='x', which='major', linestyle=":", alpha=0.3, color='#95a5a6', zorder=0)

# ==========================================
# 6. 点睛之笔：悬浮数据标签与学术故事 (Storytelling)
# ==========================================
# 精细打标：避免文字与折线重叠
for x, y in zip(prompt, feasible_ratio):
    # 根据 Y 值的不同，微调文字的位置 (偏下一点避免挡住红色阴影)
    ax.text(x, y - 2, f"{y:.1f}%", ha="center", va="top",
            fontsize=14, fontweight='bold', color='#2c3e50', zorder=4)

# 自定义图例位置与样式
ax.legend(loc="lower left", frameon=True, facecolor='white', edgecolor='none', fontsize=16)


# ==========================================
# 7. 保存与输出
# ==========================================
plt.tight_layout()
plt.savefig("fig16_feasible_ratio_vs_prompt.pdf", bbox_inches="tight")
plt.savefig("fig16_feasible_ratio_vs_prompt.png", dpi=600, bbox_inches="tight")
plt.show()