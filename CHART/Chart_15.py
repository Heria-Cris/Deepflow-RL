import matplotlib.pyplot as plt

rows = [
    ["Weak / 128", "124.60", "124.60", "P=0,K=1,MB=8", "P=0,K=1,MB=8"],
    ["Weak / 512", "39.53", "39.53", "P=0,K=1,MB=2", "P=0,K=1,MB=2"],
    ["Weak / 2048", "10.05", "10.05", "P=0,K=1,MB=1", "P=0,K=1,MB=1"],
    ["Weak / 4096", "4.70", "4.70", "P=0,K=1,MB=1", "P=0,K=1,MB=1"],
    ["Strong / 512", "41.82", "41.82", "P=0,K=1,MB=1", "P=0,K=1,MB=1"],
    ["Strong / 2048", "10.12", "10.12", "P=0,K=1,MB=1", "P=0,K=1,MB=1"],
]
cols = ["Scenario", "Oracle TP", "PPO TP", "Oracle Action", "PPO Action"]

fig, ax = plt.subplots(figsize=(11, 4))
ax.axis("off")
table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

plt.title("Best PPO vs Feasible Oracle")
plt.tight_layout()
plt.savefig("fig15_ppo_vs_oracle_table.png", dpi=300, bbox_inches="tight")
plt.savefig("fig15_ppo_vs_oracle_table.pdf", bbox_inches="tight")
plt.show()