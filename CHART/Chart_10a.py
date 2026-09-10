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

# MB values from Experiment 5
data = np.array([
    [4, 4, 4, 4],   # lat 10
    [8, 8, 8, 8],   # lat 50
    [8, 8, 8, 8],   # lat 100
])

fig, ax = plt.subplots(figsize=(6.5, 4.5))

# ==========================================
# 3. 核心突破：构建离散色彩映射 (Discrete Colormap)
# ==========================================
# 学术界经典的对比色：深蓝 (代表较小的MB) vs 砖红 (代表较大的MB)
color_4 = '#2980b9'  # Blue
color_8 = '#c0392b'  # Red
cmap = mcolors.ListedColormap([color_4, color_8])

# 设置边界，确保 4 落在第一个颜色，8 落在第二个颜色
bounds = [3, 6, 9]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# 绘制 Heatmap，aspect="equal" 保证每个格子是完美的正方形
im = ax.imshow(data, cmap=cmap, norm=norm, aspect="equal")

# ==========================================
# 4. 图表修饰：刻度与标签
# ==========================================
ax.set_xticks(np.arange(len(bw_labels)))
ax.set_xticklabels(bw_labels, fontsize=11, fontweight='bold')
ax.set_yticks(np.arange(len(lat_labels)))
ax.set_yticklabels(lat_labels, fontsize=11, fontweight='bold')

ax.set_xlabel("Network Bandwidth (Mbps)", fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel("Network Latency (ms)", fontsize=13, fontweight='bold', labelpad=10)

# ==========================================
# 5. 点睛之笔 A：添加高逼格的白色分割网格 (White Grid Lines)
# ==========================================
# 通过设置次级刻度 (minor ticks) 并在此绘制白线，切割色块
ax.set_xticks(np.arange(-.5, len(bw_labels), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(lat_labels), 1), minor=True)
ax.grid(which="minor", color="white", linestyle='-', linewidth=3) # 粗白线切割
ax.tick_params(which="minor", bottom=False, left=False) # 隐藏次级刻度线本身
ax.tick_params(which="major", length=5, width=1.5) # 加粗主刻度线

# ==========================================
# 6. 点睛之笔 B：高反差文本标注
# ==========================================
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        # 因为背景是深色（深蓝/深红），文字必须使用纯白色加粗，这是顶会的标准审美
        ax.text(j, i, f"MB={data[i, j]}",
                ha="center", va="center", color="white",
                fontsize=13, fontweight='bold')

# 取消右侧冗余的 Colorbar（因为格子里的字已经说明了一切），增加数据墨水比
# plt.colorbar(...) 被刻意移除

ax.set_title("Optimal Policy Map (Prompt = 128)", fontsize=14, fontweight='bold', pad=20)

# ==========================================
# 7. 保存与输出
# ==========================================
plt.tight_layout()
plt.savefig("fig10a_policy_map_prompt128.pdf", bbox_inches="tight")
plt.savefig("fig10a_policy_map_prompt128.png", dpi=600, bbox_inches="tight")
plt.show()