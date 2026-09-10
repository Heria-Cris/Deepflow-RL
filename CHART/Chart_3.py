import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 全局学术风设置
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# ==========================================
# 2. 新数据
# ==========================================
seq_throughput, pipe_throughput = 16.10, 18.81
seq_makespan, pipe_makespan = 8.2510, 7.0636
seq_util, pipe_util = 16.4, 19.1

metrics = [
    "Throughput\n(tok/s)",
    "Makespan\n(seconds)",
    "Cloud Util\n(%)"
]

seq_vals = [seq_throughput, seq_makespan, seq_util]
pipe_vals = [pipe_throughput, pipe_makespan, pipe_util]

x = np.arange(len(metrics))
width = 0.32

# ==========================================
# 3. 学术配色
# ==========================================
color_seq = '#95a5a6'   # 灰色
color_pipe = '#2980b9'  # 蓝色

fig, ax = plt.subplots(figsize=(8.5, 5.5))

bars_seq = ax.bar(
    x - width / 2, seq_vals, width,
    label="Sequential (Baseline)",
    color=color_seq, edgecolor='black', linewidth=1.2, zorder=3
)

bars_pipe = ax.bar(
    x + width / 2, pipe_vals, width,
    label="Pipelined (Proposed)",
    color=color_pipe, edgecolor='black', linewidth=1.2, zorder=3
)

# ==========================================
# 4. 图表修饰
# ==========================================
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=17, fontweight='bold')
ax.tick_params(axis='y', labelsize=17)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.grid(axis="y", linestyle="--", alpha=0.5, color='#bdc3c7', zorder=0)
ax.set_axisbelow(True)

ax.set_ylim(0, max(max(seq_vals), max(pipe_vals)) * 1.35)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.15),
    ncol=2,
    frameon=False,
    fontsize=17
)

# ==========================================
# 5. 数据标签与变化率
# ==========================================
for i in range(len(metrics)):
    v_seq = seq_vals[i]
    v_pipe = pipe_vals[i]

    x_seq = bars_seq[i].get_x() + bars_seq[i].get_width() / 2
    x_pipe = bars_pipe[i].get_x() + bars_pipe[i].get_width() / 2

    # 数值标签
    if i == 1:  # makespan 保留 4 位小数
        ax.text(x_seq, v_seq + 0.3, f"{v_seq:.4f}", ha="center", va="bottom", fontsize=16)
        ax.text(x_pipe, v_pipe + 0.3, f"{v_pipe:.4f}", ha="center", va="bottom", fontsize=16)
    else:
        ax.text(x_seq, v_seq + 0.3, f"{v_seq:.2f}", ha="center", va="bottom", fontsize=16)
        ax.text(x_pipe, v_pipe + 0.3, f"{v_pipe:.2f}", ha="center", va="bottom", fontsize=16)

    # 变化率
    change_pct = ((v_pipe - v_seq) / v_seq) * 100

    y_max = max(v_seq, v_pipe) + 1.8
    bracket_y = y_max + 0.2

    ax.plot(
        [x_seq, x_seq, x_pipe, x_pipe],
        [y_max, bracket_y, bracket_y, y_max],
        color='black', linewidth=1.0
    )

    # 对于 makespan，下降是好事，所以仍用绿色强调
    if i == 1:
        improvement_pct = ((v_seq - v_pipe) / v_seq) * 100
        delta_text = f"-{improvement_pct:.1f}% $\\downarrow$"
        text_color = '#27ae60'
    else:
        delta_text = f"+{change_pct:.1f}% $\\uparrow$"
        text_color = '#27ae60'

    ax.text(
        (x_seq + x_pipe) / 2, bracket_y + 0.2, delta_text,
        ha='center', va='bottom', fontsize=16, fontweight='bold', color=text_color
    )

# ==========================================
# 6. 保存与输出
# ==========================================
plt.tight_layout()
plt.savefig("fig3_pipeline_vs_sequential.pdf", bbox_inches="tight")
plt.savefig("fig3_pipeline_vs_sequential.png", dpi=600, bbox_inches="tight")
plt.show()