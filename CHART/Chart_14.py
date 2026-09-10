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
prompt = np.array([128, 512, 2048, 4096])

# Weak scenario: 1 Mbps, 50 ms
weak_tp = np.array([124.60, 39.53, 10.05, 4.70])

# Strong scenario: 100 Mbps, 10 ms
strong_tp = np.array([150.61, 41.82, 10.12, 4.71])

# ==========================================
# 3. 高级学术配色
# ==========================================
color_strong = '#2980b9'  # 强网：自信的深蓝
color_weak = '#e67e22'    # 弱网：警示的暗橘色
color_fill = '#bdc3c7'    # 填充：柔和的灰蓝

fig, ax = plt.subplots(figsize=(8.5, 5.5))

# ==========================================
# 4. 杀手锏 A：绘制差距填充带 (Gap Shading)
# ==========================================
# 这是一招绝杀：通过填充两条线之间的区域，直观展示“网络收益的边际递减”
ax.fill_between(prompt, weak_tp, strong_tp, color=color_fill, alpha=0.3, label="Network Speedup Gap")

# ==========================================
# 5. 绘制精美折线 (Hollow Markers)
# ==========================================
ax.plot(prompt, strong_tp, marker="s", markersize=8, mfc='white', mew=1.5,
        color=color_strong, linewidth=2.5, label="Strong Network (100Mbps, 10ms)", zorder=3)

ax.plot(prompt, weak_tp, marker="o", markersize=8, mfc='white', mew=1.5,
        color=color_weak, linewidth=2.5, linestyle="--", label="Weak Network (1Mbps, 50ms)", zorder=3)

# ==========================================
# 6. 核心修复：X轴对数化 (Log-2 Scale)
# ==========================================
ax.set_xscale('log', base=2)
# 强制设定刻度为给定的 prompt 值，并显示为普通数字
ax.set_xticks(prompt)
ax.set_xticklabels([str(p) for p in prompt], fontsize=17, fontweight='bold')
ax.tick_params(axis='y', labelsize=17)

# ==========================================
# 7. 图表修饰 (Data-Ink Ratio 优化)
# ==========================================
ax.set_xlabel("Prompt Length (Log Scale)", fontsize=17, fontweight='bold', labelpad=10)
ax.set_ylabel("Throughput (tok/s)", fontsize=17, fontweight='bold', labelpad=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.grid(axis='y', linestyle="--", alpha=0.4, color='#95a5a6', zorder=0)
ax.grid(axis='x', which='major', linestyle=":", alpha=0.3, color='#95a5a6', zorder=0)

ax.set_ylim(0, 175) # 留出顶部空间

# 图例放置在右上角内部，去边框
ax.legend(loc="upper right", frameon=False, fontsize=17)

# 局部微调数值标签，避免遮挡
ax.text(128, strong_tp[0] + 4, f"{strong_tp[0]:.1f}", ha='center', va='bottom', fontsize=15, color=color_strong, fontweight='bold')
ax.text(128, weak_tp[0] - 8, f"{weak_tp[0]:.1f}", ha='center', va='top', fontsize=15, color=color_weak, fontweight='bold')

ax.set_title("Throughput Degradation & Bottleneck Convergence", fontsize=14, fontweight='bold', pad=20)

# ==========================================
# 9. 保存与输出
# ==========================================
plt.tight_layout()
plt.savefig("fig14_throughput_vs_prompt.pdf", bbox_inches="tight")
plt.savefig("fig14_throughput_vs_prompt.png", dpi=600, bbox_inches="tight")
plt.show()