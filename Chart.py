import matplotlib.pyplot as plt
import numpy as np


def plot_paper_chart():
    # 数据准备
    scenarios = [
        'Prefill: Local Compute',
        'Prefill: Net Transfer',
        'Decode: Edge Memory IO',
        'Decode: Legacy Offload',
        'Decode: DeepFlow (Ours)'
    ]

    # 转换为毫秒 (ms)
    times = [
        2.49,  # Prefill Local
        32050.00,  # Prefill Network (32.05s)
        9.20,  # Decode Edge IO
        362.50,  # Legacy
        50.20  # Ours
    ]

    colors = ['#d3d3d3', '#ff6b6b', '#d3d3d3', '#feca57', '#1dd1a1']
    hatches = ['', 'xx', '//', '..', '**']

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制柱状图
    bars = ax.barh(scenarios, times, color=colors, edgecolor='black', alpha=0.9)

    # 设置对数坐标轴 (因为差异高达 10000倍)
    ax.set_xscale('log')

    # 添加纹理
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    # 添加具体数值标签
    for i, v in enumerate(times):
        label = f"{v:.2f} ms"
        if v > 1000:
            label = f"{v / 1000:.2f} s"
        ax.text(v * 1.1, i, label, va='center', fontweight='bold', fontsize=10)

    # 辅助线：物理 RTT
    plt.axvline(x=50.0, color='blue', linestyle='--', linewidth=1.5, label='Physical RTT limit (50ms)')

    # 美化图表
    ax.set_xlabel('Latency (ms) - Log Scale', fontsize=12, fontweight='bold')
    ax.set_title('Latency Breakdown & Bottleneck Analysis (1Mbps Network)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # 让 Prefill 在最上面
    ax.legend(loc='lower right')

    # 添加区域注释
    plt.text(40000, 0.8, 'Communication Wall\n(Bandwidth Bound)', color='red', ha='left', va='center', fontsize=9,
             fontweight='bold')
    plt.text(180, 4, 'Theoretical Limit\n(Latency Bound)', color='green', ha='left', va='center', fontsize=9,
             fontweight='bold')

    plt.tight_layout()
    plt.savefig('bottleneck_analysis.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_paper_chart()