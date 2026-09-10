import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
events = [
    # mb_id, device, type, start, end
    (0, "Jetson",  "Draft",  0.0000, 0.8554),
    (0, "Network", "Comm",   0.8554, 0.9067),
    (1, "Jetson",  "Draft",  0.8554, 1.7109),
    (0, "A100",    "Verify", 0.9067, 1.0756),
    (1, "Network", "Comm",   1.7109, 1.7621),
    (2, "Jetson",  "Draft",  1.7109, 2.5663),
    (1, "A100",    "Verify", 1.7621, 1.9310),
    (2, "Network", "Comm",   2.5663, 2.6175),
    (2, "A100",    "Verify", 2.6175, 2.7864),
    (7, "Jetson",  "Draft",  5.9880, 6.8435),
    (7, "Network", "Comm",   6.8435, 6.8947),
    (7, "A100",    "Verify", 6.8947, 7.0636),
]

# ==========================================
# 3. 配色与坐标映射
# ==========================================
color_map = {
    "Draft": "#3498db",
    "Comm": "#f39c12",
    "Verify": "#2ecc71"
}

y_map = {"A100": 0, "Network": 1, "Jetson": 2}
y_labels = {
    "A100": "A100\n(Verify)",
    "Network": "Network\n(Comm)",
    "Jetson": "Jetson\n(Draft)"
}
box_height = 0.65

fig, ax = plt.subplots(figsize=(10.5, 5))
added_labels = set()

# ==========================================
# 4. 绘制流水线块
# ==========================================
for mb_id, device, typ, start, end in events:
    y_center = y_map[device]
    width = end - start
    y_bottom = y_center - box_height / 2

    c = color_map[typ]
    label = typ if typ not in added_labels else ""
    added_labels.add(typ)

    rect = patches.Rectangle(
        (start, y_bottom), width, box_height,
        facecolor=c, edgecolor='black', linewidth=1.0,
        label=label, zorder=3
    )
    ax.add_patch(rect)

    # 只在较长块内部写 MB 标签
    if typ != "Comm":
        ax.text(
            start + width / 2, y_center, f"MB {mb_id}",
            ha="center", va="center",
            fontsize=10, fontweight='bold', color='black'
        )

# ==========================================
# 5. 绘制依赖箭头
# ==========================================
arrow_props = dict(
    facecolor='black', edgecolor='black',
    arrowstyle='-|>', mutation_scale=10,
    lw=1.2, alpha=0.8
)

# MB0: Jetson -> Network
ax.annotate(
    '',
    xy=(0.8554, y_map["Network"] + box_height / 2 + 0.05),
    xytext=(0.8554, y_map["Jetson"] - box_height / 2 - 0.05),
    arrowprops=arrow_props
)

# MB0: Network -> A100
ax.annotate(
    '',
    xy=(0.9067, y_map["A100"] + box_height / 2 + 0.05),
    xytext=(0.9067, y_map["Network"] - box_height / 2 - 0.05),
    arrowprops=arrow_props
)

# MB1: Jetson -> Network
ax.annotate(
    '',
    xy=(1.7109, y_map["Network"] + box_height / 2 + 0.05),
    xytext=(1.7109, y_map["Jetson"] - box_height / 2 - 0.05),
    arrowprops=arrow_props
)

# 可选：MB1 Network -> A100
ax.annotate(
    '',
    xy=(1.7621, y_map["A100"] + box_height / 2 + 0.05),
    xytext=(1.7621, y_map["Network"] - box_height / 2 - 0.05),
    arrowprops=arrow_props
)

# ==========================================
# 6. 省略的微批次标记
# ==========================================
ax.text(
    4.35, 1, ". . .   Omitted MB 3 - MB 6   . . .",
    ha='center', va='center',
    fontsize=12, style='italic', color='#7f8c8d'
)

# ==========================================
# 7. 图表修饰
# ==========================================
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(
    [y_labels["A100"], y_labels["Network"], y_labels["Jetson"]],
    fontsize=11.5, fontweight='bold'
)

ax.set_xlabel("Timeline (seconds)", fontsize=13, fontweight='bold', labelpad=10)
ax.tick_params(axis='x', labelsize=11)

ax.set_xlim(-0.2, 7.5)
ax.set_ylim(-0.8, 2.8)

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.15),
    ncol=3,
    frameon=False,
    fontsize=12
)

ax.grid(axis='x', linestyle='--', alpha=0.5, color='#bdc3c7', zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.tick_params(axis='y', which='both', length=0)

ax.set_title(
    "Pipeline Execution Timeline on Edge-Cloud System",
    fontsize=14, pad=32, fontweight='bold'
)

# ==========================================
# 8. 保存与输出
# ==========================================
plt.tight_layout()
plt.savefig("fig4_pipeline_gantt.pdf", bbox_inches="tight")
plt.savefig("fig4_pipeline_gantt.png", dpi=600, bbox_inches="tight")
plt.show()