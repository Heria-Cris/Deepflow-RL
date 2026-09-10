import matplotlib.pyplot as plt

rows = [
    ["Weak / 128", "P=0, K=1, MB=8", "124.60", "N/A"],
    ["Weak / 512", "P=0, K=1, MB=2", "39.53", "Yes"],
    ["Weak / 2048", "P=0, K=1, MB=1", "10.05", "Yes"],
    ["Moderate / 512", "P=0, K=1, MB=1", "41.23", "No"],
    ["Strong / 512", "P=0, K=1, MB=1", "41.82", "No"],
    ["Strong / 2048", "P=0, K=1, MB=1", "10.12", "No"],
    ["Very Weak / 512", "P=0, K=1, MB=2", "38.75", "Yes"],
    ["High RTT / 512", "P=0, K=1, MB=4", "35.85", "Yes"],
]
cols = ["Scenario", "Chosen Action", "Throughput", "Changed?"]

fig, ax = plt.subplots(figsize=(10.5, 4.2))
ax.axis("off")
table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

plt.title("Action Sensitivity Analysis")
plt.tight_layout()
plt.savefig("fig9_action_sensitivity_table.png", dpi=300, bbox_inches="tight")
plt.savefig("fig9_action_sensitivity_table.pdf", bbox_inches="tight")
plt.show()