import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# ==========================================
# 1. 全局学术风设置
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.5

# ==========================================
# 2. 数据准备
# ==========================================
bw_labels = ["0.5", "1", "10", "100"]
lat_labels = ["10", "50", "100"]

# (a) Prompt=128
data_128 = np.array([
    [4, 4, 4, 4],
    [8, 8, 8, 8],
    [8, 8, 8, 8],
])

# (b) Prompt=512
data_512 = np.array([
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [4, 4, 4, 4],
])

# (c) Prompt=2048
data_2048 = np.array([
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
])

datasets = [data_128, data_512, data_2048]
titles = ["(a) Prompt = 128", "(b) Prompt = 512", "(c) Prompt = 2048"]

# ==========================================
# 3. 终极跨图一致性：全局离散色彩系统 (Global Semantic Palette)
# ==========================================
# 定义囊括所有可能 MB 值 (1, 2, 4, 8) 的调色盘
color_1 = '#16a085'  # 翡翠绿 (MB=1)
color_2 = '#8e44ad'  # 深紫色 (MB=2)
color_4 = '#2980b9'  # 学术蓝 (MB=4)
color_8 = '#c0392b'  # 砖红色 (MB=8)

cmap = mcolors.ListedColormap([color_1, color_2, color_4, color_8])

# 严格的数值边界映射：
# [0.5, 1.5) -> Green (1)
# [1.5, 3.5) -> Purple (2)
# [3.5, 6.0) -> Blue (4)
# [6.0, 9.5) -> Red (8)
bounds = [0.5, 1.5, 3.5, 6.0, 9.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# ==========================================
# 4. 绘制 1x3 组图 (共享 Y 轴)
# ==========================================
# sharey=True 是学术组图的灵魂，去掉中间和右边图的 Y 轴标签，极其清爽
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

for idx, ax in enumerate(axes):
    data = datasets[idx]

    # 绘制 Heatmap
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="equal")

    # 刻度设置
    ax.set_xticks(np.arange(len(bw_labels)))
    ax.set_xticklabels(bw_labels, fontsize=15, fontweight='bold')
    # 因为 sharey=True，只需要对第一个子图设置 Y 轴刻度即可
    if idx == 0:
        ax.set_yticks(np.arange(len(lat_labels)))
        ax.set_yticklabels(lat_labels, fontsize=15, fontweight='bold')
        ax.set_ylabel("Network Latency (ms)", fontsize=17, fontweight='bold', labelpad=10)

    ax.set_xlabel("Bandwidth (Mbps)", fontsize=17, fontweight='bold', labelpad=10)
    ax.set_title(titles[idx], fontsize=17, fontweight='bold', pad=15)

    # ==========================================
    # 5. 棋盘格切割与白字标注
    # ==========================================
    # 粗白线切割
    ax.set_xticks(np.arange(-.5, len(bw_labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(lat_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=5, width=1.5)

    # 填充高对比白色文本
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"MB={data[i, j]}",
                    ha="center", va="center", color="white",
                    fontsize=14, fontweight='bold')


# ==========================================
# 7. 保存与输出
# ==========================================
# 调整子图间距，让它们紧密靠拢
plt.subplots_adjust(wspace=0.05)
plt.savefig("fig10_combined_policy_map.pdf", bbox_inches="tight")
plt.savefig("fig10_combined_policy_map.png", dpi=600, bbox_inches="tight")
plt.show()