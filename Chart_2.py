import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_motivation_figure():
    # 设置 IEEE 风格字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 12

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ==========================================
    # 子图 1: 性能崩塌对比 (Performance Collapse)
    # ==========================================
    strategies = ['Sequential\n(Batching)', 'Pipelined\n(Micro-batching)']
    throughput = [1309.0, 344.9]
    utilization = [13.0, 3.4]

    # 双 Y 轴
    ax1_r = ax1.twinx()

    # 绘制柱状图
    width = 0.35
    x = range(len(strategies))
    b1 = ax1.bar([i - width / 2 for i in x], throughput, width, label='Throughput', color='#2c3e50', alpha=0.9)
    b2 = ax1_r.bar([i + width / 2 for i in x], utilization, width, label='Cloud GPU Util.', color='#e74c3c', alpha=0.9,
                   hatch='//')

    # 设置标签
    ax1.set_ylabel('Throughput (tokens/s)', color='#2c3e50', fontweight='bold')
    ax1_r.set_ylabel('Cloud Utilization (%)', color='#e74c3c', fontweight='bold')
    ax1.set_title('(a) Performance Degradation', fontweight='bold', y=-0.20)

    # 添加数值标签
    for i, v in enumerate(throughput):
        ax1.text(i - width / 2, v + 20, f"{int(v)}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    for i, v in enumerate(utilization):
        ax1_r.text(i + width / 2, v + 0.2, f"{v}%", ha='center', va='bottom', fontsize=10, color='#c0392b')

    # 标注 "Negative Speedup"
    ax1.annotate('Granularity Trap:\n0.26x Speedup',
                 xy=(1, 350), xytext=(0.5, 800),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                 fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

    # ==========================================
    # 子图 2: 甘特图揭示原因 (Mechanism Analysis)
    # ==========================================
    # 数据来自日志
    # (Start, Duration)
    # MB 0
    edge_0 = (0.0000, 0.0514)
    net_0 = (0.0514, 0.0506)  # 0.1020 - 0.0514
    cloud_0 = (0.1020, 0.0020)  # 0.1040 - 0.1020
    # MB 1
    edge_1 = (0.0514, 0.0514)
    net_1 = (0.1028, 0.0506)
    cloud_1 = (0.1534, 0.0020)
    # MB 2
    edge_2 = (0.1028, 0.0514)
    net_2 = (0.1542, 0.0506)
    cloud_2 = (0.2048, 0.0020)

    # 绘制 Gantt
    yticks = [3, 2, 1]
    ylabels = ['Cloud (Verify)', 'Network (Comm)', 'Edge (Draft)']
    colors = {'Cloud': '#27ae60', 'Network': '#e74c3c', 'Edge': '#3498db'}

    # Helper function
    def plot_bar(y, interval, color):
        ax2.broken_barh([interval], (y - 0.4, 0.8), facecolors=color, edgecolor='black')

    # MB 0
    plot_bar(1, edge_0, colors['Edge'])
    plot_bar(2, net_0, colors['Network'])
    plot_bar(3, cloud_0, colors['Cloud'])

    # MB 1
    plot_bar(1, edge_1, colors['Edge'])
    plot_bar(2, net_1, colors['Network'])
    plot_bar(3, cloud_1, colors['Cloud'])

    # MB 2
    plot_bar(1, edge_2, colors['Edge'])
    plot_bar(2, net_2, colors['Network'])
    plot_bar(3, cloud_2, colors['Cloud'])

    # 设置轴
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(ylabels)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_title('(b) Timeline Zoom-in (The "Stall" Phenomenon)', fontweight='bold', y=-0.20)
    ax2.set_xlim(0, 0.25)
    ax2.grid(True, axis='x', linestyle='--', alpha=0.5)

    # 添加注释：指出网络是瓶颈
    ax2.text(0.075, 2, 'RTT Dominant\n(~50ms)', color='white', ha='center', va='center', fontsize=9, fontweight='bold')

    # 添加注释：指出云端空闲
    ax2.annotate('Resource Starvation\n(Idle Gaps)',
                 xy=(0.12, 3), xytext=(0.16, 3.3),
                 arrowprops=dict(arrowstyle='->', lw=1.5),
                 fontsize=10, color='#c0392b', fontweight='bold')

    # Legend
    patches = [mpatches.Patch(color=v, label=k) for k, v in colors.items()]
    ax2.legend(handles=patches, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig('granularity_trap.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_motivation_figure()