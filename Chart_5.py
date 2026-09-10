import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 实验数据录入
# ==========================================
bandwidths = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]

# Legacy Split (Baseline) - 随带宽线性增长
legacy_data = [7.84, 15.37, 29.56, 66.31, 113.25, 261.08, 311.99]

# DeepFlow-RL (Ours) - 带宽免疫，极其平稳
deepflow_data = [1032.20, 1081.36, 1107.74, 1124.20, 1129.79, 1134.31, 1134.87]

# ==========================================
# 2. 全局绘图设置 (IEEE 风格)
# ==========================================
plt.rcParams['font.family'] = 'serif'  # 使用衬线字体 (Times New Roman风格)
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5   # 坐标轴加粗

# 创建画布 (宽高比 4:3 或 16:9 适合论文排版)
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# ==========================================
# 3. 绘制曲线
# ==========================================

# 绘制 DeepFlow-RL (Ours) - 红色实线，星号标记
ax.plot(bandwidths, deepflow_data,
        label='DeepFlow-RL (Ours)',
        color='#D62728',  # 砖红色
        marker='*',       # 星号
        markersize=10,
        linewidth=2.5,
        linestyle='-')

# 绘制 Legacy Split (Baseline) - 蓝色虚线，三角标记
ax.plot(bandwidths, legacy_data,
        label='Act Split (Baseline)',
        color='#1F77B4',  # 深蓝色
        marker='^',       # 三角
        markersize=8,
        linewidth=2.0,
        linestyle='--')

# ==========================================
# 4. 坐标轴与刻度调整
# ==========================================

# X轴设置对数坐标 (Log Scale) - 关键！
ax.set_xscale('log')

# 手动设置 X 轴刻度，避免对数坐标显示不直观
ax.set_xticks(bandwidths)
ax.set_xticklabels([str(b) for b in bandwidths])

# 设置 Y 轴范围 (留出一点顶部空间放图例)
ax.set_ylim(0, 1400)

# 设置标签
ax.set_xlabel('Network Bandwidth (Mbps) [Log Scale]', fontweight='bold')
ax.set_ylabel('Throughput (tokens/s)', fontweight='bold')
ax.set_title('Sensitivity to Network Bandwidth', fontweight='bold', pad=15)

# ==========================================
# 5. 添加标注与细节 (Storytelling)
# ==========================================

# 添加网格线 (Grid)
ax.grid(True, which="both", ls="-", alpha=0.2)

# 标注关键差距 (131x Speedup)
# 在 0.5 Mbps 处画一条双向箭头
ax.annotate('', xy=(0.5, 1032), xytext=(0.5, 7.8),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax.text(0.55, 500, '131x Speedup\n(Communication Wall)',
        fontsize=10, color='black', verticalalignment='center')

# 标注 "Bandwidth Immunity" 特性
ax.text(5, 1180, 'Bandwidth Immunity Zone',
        fontsize=10, color='#D62728', fontweight='bold')

# 添加图例
ax.legend(loc='lower right', frameon=True, shadow=True, fancybox=True)

# ==========================================
# 6. 保存与展示
# ==========================================
plt.tight_layout()
plt.savefig('bandwidth_sensitivity.png', dpi=300) # 保存高清图
plt.savefig('bandwidth_sensitivity.pdf', format='pdf') # 保存矢量图 (推荐插入论文)
plt.show()