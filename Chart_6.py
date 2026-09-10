import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 数据准备
# ==========================================
scenarios = ['Fixed Pipeline\n(Heuristic A)', 'Fixed No-Spec\n(Heuristic B)', 'DeepFlow-RL\n(Ours)']
throughput = [220.61, 559.34, 1081.36]
details = ['MB=4, K=10\n(Granularity Trap)', 'MB=32, K=0\n(Latency Bound)', 'MB=32, K=7\n(Global Optimal)']

# 颜色设置：灰色用于 Baseline，醒目的红色用于 Ours
colors = ['#A9A9A9', '#808080', '#D62728'] # 浅灰，深灰，砖红

# ==========================================
# 2. 全局绘图设置 (IEEE 风格)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# ==========================================
# 3. 绘制柱状图
# ==========================================
bars = ax.bar(scenarios, throughput, color=colors, width=0.6, edgecolor='black', linewidth=1)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 15,
            f'{height:.1f}',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

# ==========================================
# 4. 添加策略详情标注 (放在柱子内部或下方)
# ==========================================
# 这里我们将参数详情写在 X 轴标签里了，所以不需要额外标注

# ==========================================
# 5. 添加对比箭头与解释 (Storytelling)
# ==========================================

# 箭头 1: Pipeline vs Ours (4.9x)
# 从第一个柱子顶端指向第三个柱子腰部
ax.annotate('Avoids Negative Speedup\n(4.9x vs Pipeline)',
            xy=(2, 1081), xytext=(0, 400),
            arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=-0.2", color='blue', lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.9),
            ha='center', color='blue', fontsize=10)

# 箭头 2: No-Spec vs Ours (1.9x)
# 从第二个柱子顶端指向第三个柱子顶端
ax.annotate('Speculation Gain\n(1.9x vs No-Spec)',
            xy=(2, 1081), xytext=(1, 700),
            arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=0.2", color='red', lw=1.5),
            ha='center', color='#D62728', fontweight='bold', fontsize=10)

# ==========================================
# 6. 坐标轴调整
# ==========================================
ax.set_ylabel('Throughput (tokens/s)', fontweight='bold')
ax.set_title('Ablation Study: Impact of RL & Speculation\n(Weak Network: 1Mbps, 50ms)', fontweight='bold', pad=15)
ax.set_ylim(0, 1400) # 留出顶部空间画箭头

# 添加网格
ax.grid(axis='y', linestyle='--', alpha=0.5)

# ==========================================
# 7. 保存
# ==========================================
plt.tight_layout()
plt.savefig('ablation_study_bar.png', dpi=300)
plt.savefig('ablation_study_bar.pdf', format='pdf')
plt.show()