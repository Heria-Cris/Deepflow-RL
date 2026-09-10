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

# MB values from Experiment 5 (Prompt=2048)
# All values are 1, representing a massive compute-bound scenario
data = np.array([
    [1, 1, 1, 1],   # lat 10
    [1, 1, 1, 1],   # lat 50
    [1, 1, 1, 1],   # lat 100
])

fig, ax = plt.subplots(figsize=(6.5, 4.5))

# ==========================================
# 3. 跨图一致的离散色彩系统 (Semantic Color Locking)
# ==========================================
# 强制锁定：MB=1 必须与上一张图保持完全相同的“翡翠绿”
color_1 = '#16a085'

# 因为数据全是1，为了防止 imshow 报错或默认渐变，我们显式指定单色色板和边界
cmap = mcolors.ListedColormap([color_1])
bounds = [0.5, 1.5]
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
# 如果全是绿色没有白线，整张图会变成一个死板的纯色大方块。
# 粗白线能保留“矩阵/棋盘”的质感，体现出系统对 12 种不同环境都做出了独立决策。
ax.set_xticks(np.arange(-.5, len(bw_labels), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(lat_labels), 1), minor=True)
ax.grid(which="minor", color="white", linestyle='-', linewidth=3)
ax.tick_params(which="minor", bottom=False, left=False)
ax.tick_params(which="major", length=5, width=1.5)

# ==========================================
# 6. 点睛之笔 B：高反差文本标注
# ==========================================
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        # 纯白、加粗的大号文字，极具学术张力
        ax.text(j, i, f"MB={data[i, j]}",
                ha="center", va="center", color="white",
                fontsize=13, fontweight='bold')

# 废弃 Colorbar，最大化数据墨水比

ax.set_title("Optimal Policy Map (Prompt = 2048)", fontsize=14, fontweight='bold', pad=20)

# ==========================================
# 7. 保存与输出
# ==========================================
plt.tight_layout()
plt.savefig("fig10c_policy_map_prompt2048.pdf", bbox_inches="tight")
plt.savefig("fig10c_policy_map_prompt2048.png", dpi=600, bbox_inches="tight")
plt.show()