import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 数据准备
# ==========================================
labels = ['Weak Network\n(1Mbps, 50ms)', 'Strong Network\n(100Mbps, 10ms)']

# 数据集
Local_Only = [8.8, 16.8]
Act_Split = [15.4, 511.45]
Cloud_Native = [559.3, 1969.97]
Deepflow = [1081.36, 1604.04]

# 设置位置
x = np.arange(len(labels))
width = 0.2  # 柱状图宽度

# ==========================================
# 2. 创建图表
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

# 颜色配置：灰色系用于Baseline，鲜艳颜色用于Ours
colors = ['#d9d9d9', '#bdbdbd', '#8da0cb', '#fc8d62']
# 边框颜色
edge_color = 'black'

# 绘制柱子
rects1 = ax.bar(x - 1.5*width, Local_Only, width, label='Local Only', color=colors[0], edgecolor=edge_color)
rects2 = ax.bar(x - 0.5*width, Act_Split, width, label='Act Split', color=colors[1], edgecolor=edge_color, hatch='//')
rects3 = ax.bar(x + 0.5*width, Cloud_Native, width, label='Cloud Native', color=colors[2], edgecolor=edge_color)
rects4 = ax.bar(x + 1.5*width, Deepflow, width, label='DeepFlow-RL (Ours)', color=colors[3], edgecolor=edge_color, zorder=10)

# ==========================================
# 3. 样式美化与标注
# ==========================================

# 设置 Y 轴为对数坐标 (关键步骤！)
ax.set_yscale('log')
# 设置 Y 轴范围，留出头部空间给标签
ax.set_ylim(5, 8000)

# 设置标签和标题
ax.set_ylabel('Throughput (tokens/s) - Log Scale', fontsize=14, fontweight='bold')
ax.set_title('Performance Comparison under Different Network Conditions', fontsize=16, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
ax.legend(fontsize=11, loc='upper left', frameon=True, fancybox=True, shadow=True)

# 添加网格线 (仅 Y 轴)
ax.grid(axis='y', linestyle='--', alpha=0.6, which='major')

# --- 自动添加数值标签函数 ---
def autolabel(rects, is_bold=False):
    for rect in rects:
        height = rect.get_height()
        # 对于 Log Scale，位置需要稍微调整
        text_weight = 'bold' if is_bold else 'normal'
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 垂直偏移 5 points
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight=text_weight)

# 给每个柱子加标签
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4, is_bold=True) # 让 DeepFlow 的数字加粗

# --- 添加 Speedup 标注 (亮点) ---
# 计算弱网下的加速比 (DeepFlow / Cloud Only)
weak_speedup = Deepflow[0] / Cloud_Native[0]
# 在弱网两根柱子之间画箭头
# 获取 Cloud Only 和 DeepFlow 在弱网(index 0)的柱子对象
cloud_bar = rects3[0]
deep_bar = rects4[0]

# 箭头起始和结束点
x_start = cloud_bar.get_x() + cloud_bar.get_width()/2
x_end = deep_bar.get_x() + deep_bar.get_width()/2
y_start = Cloud_Native[0]
y_end = Deepflow[0]

# 绘制标注线
ax.annotate(f'{weak_speedup:.1f}x Speedup',
            xy=(x_start, y_start),
            xytext=(x_end + 0.15, y_end), # 文本位置稍微偏右
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.3", color='red', lw=1.5),
            fontsize=12, color='red', fontweight='bold', ha='left')

plt.tight_layout()

# 保存或显示
# plt.savefig('experiment_result.pdf', dpi=300) # 论文建议存为 PDF
plt.show()