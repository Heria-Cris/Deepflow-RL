import matplotlib.pyplot as plt

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
bw = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
legacy = [5.85, 10.55, 17.64, 22.95, 24.52, 24.75, 24.78]
static_df = [39.52, 39.53, 39.53, 39.53, 39.53, 39.53, 39.53]
ppo = [39.52, 39.53, 39.53, 39.53, 39.53, 39.53, 39.53]

# 配色方案 (学术界经典三原色变体)
color_legacy = '#7f8c8d'  # 灰色 (作为被吊打的 Baseline)
color_static = '#3498db'  # 浅蓝色 (作为理论上限/基准线)
color_ppo = '#c0392b'     # 深红色 (核心突出的 Ours)

fig, ax = plt.subplots(figsize=(8.5, 5.5))

# ==========================================
# 3. 核心修复：X轴对数化 (Log Scale)
# ==========================================
ax.set_xscale('log')

# ==========================================
# 4. 解决100%重叠：巧妙的“光晕”与“空心”画法
# ==========================================
# 技巧A：将完全一致的 Static 设为粗、半透明的底层线，作为“理论上限通道 (Oracle Upper Bound)”
ax.plot(bw, static_df, linestyle="-", linewidth=6, alpha=0.3, color=color_static,
        label="Static DeepFlow", zorder=1)

# 技巧B：将你的 PPO 画成细实线，放在 Static 上方，配合极其精致的空心大 Marker (LaTeX风)
ax.plot(bw, ppo, marker="^", markersize=9, mfc='white', mew=1.5, linewidth=2,
        color=color_ppo, label="PPO DeepFlow", zorder=3)

# 画 Legacy (灰色的普通空心圆)
ax.plot(bw, legacy, marker="o", markersize=8, mfc='white', mew=1.5, linewidth=2,
        color=color_legacy, label="Legacy Split Baseline", zorder=2)

# ==========================================
# 5. 图表修饰与讲故事 (Storytelling Annotations)
# ==========================================
# X 轴刻度设置：强制在原始数据点上打刻度，并显示为普通数字而非科学计数法
ax.set_xticks(bw)
ax.set_xticklabels([f"{b:g}" for b in bw], fontsize=17)
ax.tick_params(axis='y', labelsize=17)

ax.set_xlabel("Network Bandwidth (Mbps, Log Scale)", fontsize=17, fontweight='bold', labelpad=10)
ax.set_ylabel("Throughput (tok/s)", fontsize=17, fontweight='bold', labelpad=10)

# 去除右侧和顶部边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加次级网格线 (对数轴特有质感)
ax.grid(axis='y', linestyle="--", alpha=0.5, color='#bdc3c7', zorder=0)
ax.grid(axis='x', which='major', linestyle=":", alpha=0.3, color='#bdc3c7', zorder=0)

# 解释为什么灰线前期暴跌
ax.annotate("Network\nBottlenecked",
            xy=(1, 10.55), xytext=(3, 6),
            arrowprops=dict(facecolor=color_legacy, edgecolor=color_legacy, arrowstyle='->'),
            color=color_legacy, fontsize=15, fontweight='bold', ha='center')

# ==========================================
# 7. 图例与保存
# ==========================================
ax.legend(loc="lower right", frameon=False, fontsize=17)

plt.tight_layout()
plt.savefig("fig8_bandwidth_sensitivity.pdf", bbox_inches="tight")
plt.savefig("fig8_bandwidth_sensitivity.png", dpi=600, bbox_inches="tight")
plt.show()