import matplotlib.pyplot as plt

rows = [
    ["Weak / Short Prompt", "0.132", "0.361"],
    ["Moderate / Short Prompt", "0.129", "0.354"],
    ["Strong / Short Prompt", "0.129", "0.354"],
    ["Weak / Long Prompt", "0.137", "0.377"],
    ["Strong / Long Prompt", "0.137", "0.377"],
    ["Weak / Very Long Prompt", "0.149", "0.407"],
    ["Strong / Very Long Prompt", "0.149", "0.407"],
    ["High RTT / Short Prompt", "0.137", "0.377"],
    ["High RTT / Very Long Prompt", "0.149", "0.407"],
]
cols = ["Scenario", "Edge Peak/Budget", "Cloud Peak/Budget"]

fig, ax = plt.subplots(figsize=(9.5, 4.5))
ax.axis("off")
table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.4)

plt.title("PPO Memory Ratio Summary")
plt.tight_layout()
plt.savefig("fig12_ppo_memory_ratio_table.png", dpi=300, bbox_inches="tight")
plt.savefig("fig12_ppo_memory_ratio_table.pdf", bbox_inches="tight")
plt.show()