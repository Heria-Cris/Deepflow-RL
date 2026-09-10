import matplotlib.pyplot as plt
import numpy as np

methods = ['Local Only', 'Legacy Split', 'Cloud Only', 'DeepFlow-RL']
throughput = [8.75, 15.37, 559.34, 1223.59]
colors = ['#cccccc', '#999999', '#4c72b0', '#c44e52'] # 灰色代表基准，红/蓝代表云端/Ours

plt.figure(figsize=(10, 6))
bars = plt.bar(methods, throughput, color=colors, width=0.6)

# 添加数值标签
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 10, f'{yval:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.ylabel('Throughput (tokens/s) - Log Scale', fontsize=14)
plt.yscale('log') # 关键：使用对数坐标，否则前两项看不见
plt.title('End-to-End Throughput Comparison in Weak Network (1Mbps)', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加 Speedup 注解
plt.text(3, 1900, '140x Speedup', ha='center', color='red', fontweight='bold', fontsize=12)

plt.show()