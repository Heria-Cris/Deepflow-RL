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

# MB values from Experiment 5 (Prompt=512)
data = np.array([
    [1, 1, 1, 1],   # lat 10
    [2, 2, 2, 2],   # lat 50
    [4, 4, 4, 4],   # lat 100
])

fig, ax = plt.subplots(figsize=(6.5, 4.5))

# ==========================================
# 3. 跨图一致的离散色彩系统 (Discrete Colormap)
# ==========================================
# 保持跨图语义一致：上图中 MB=4 是蓝色，这里必须保持蓝色！
color_1 = '#16a085'  # 翡翠绿 (代表极小的 MB=1)
color_2 = '#8e44ad'  # 深紫色 (代表中等的 MB=2)
color_4 = '#2980b9'  # 学术蓝 (锁定 MB=4，与上一张图保持绝对一致)

cmap = mcolors.ListedColormap([color_1, color_2, color_4])

# 构建严格的数值边界映射：
# [0.5, 1.5) 映射为 color_1 (对应数值 1)
# [1.5, 3.5) 映射为 color_2 (对应数值 2)
# [3.5, 5.5) 映射为 color_4 (对应数值 4)
bounds = [0.5, 1.5, 3.5, 5.5]
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
        # 在深色背景上打上纯白、加粗的大号文字，极具学术张力
        ax.text(j, i, f"MB={data[i, j]}",
                ha="center", va="center", color="white",
                fontsize=13, fontweight='bold')

# 取消右侧冗余的 Colorbar (离散块状图无需渐变色带)

ax.set_title("Optimal Policy Map (Prompt = 512)", fontsize=14, fontweight='bold', pad=20)

# ==========================================
# 7. 保存与输出
# ==========================================
plt.tight_layout()
plt.savefig("fig10b_policy_map_prompt512.pdf", bbox_inches="tight")
plt.savefig("fig10b_policy_map_prompt512.png", dpi=600, bbox_inches="tight")
plt.show()