import matplotlib.pyplot as plt

rows = [
    ["Weak Net / Short Prompt", "39.53", "39.53", "P=0, K=1, MB=2"],
    ["Moderate Net / Short Prompt", "41.23", "41.23", "P=0, K=1, MB=1"],
    ["Strong Net / Short Prompt", "41.82", "41.82", "P=0, K=1, MB=1"],
    ["Weak Net / Long Prompt", "10.05", "10.05", "P=0, K=1, MB=1"],
    ["Strong Net / Long Prompt", "10.12", "10.12", "P=0, K=1, MB=1"],
]
cols = ["Scenario", "Best Static DeepFlow", "Best PPO", "PPO Action"]

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.axis("off")
table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

plt.title("Cross-Scenario Generalization")
plt.tight_layout()
plt.savefig("fig7_cross_scenario_generalization.png", dpi=300, bbox_inches="tight")
plt.savefig("fig7_cross_scenario_generalization.pdf", bbox_inches="tight")
plt.show()