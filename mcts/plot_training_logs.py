import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

with open("training_logs.json", "r") as f:
    data = json.load(f)

# ── helper ────────────────────────────────────────────────────────────────────
def plot(ax, values, title, ylabel, color, x_values=None):
    if not values:
        ax.set_title(f"{title} (no data)")
        return
    xs = x_values if x_values is not None else list(range(1, len(values) + 1))
    ax.plot(xs, values, color=color, linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Iteration" if x_values is None else "Eval #")
    ax.grid(True, alpha=0.4)

# ── layout ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
fig.suptitle("Training Logs", fontsize=15, fontweight="bold")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

ax_total   = fig.add_subplot(gs[0, 0])
ax_policy  = fig.add_subplot(gs[0, 1])
ax_value   = fig.add_subplot(gs[1, 0])
ax_winrate = fig.add_subplot(gs[1, 1])

plot(ax_total,   data.get("loss", []),          "Total Loss",   "Loss",     "#e05c5c")
plot(ax_policy,  data.get("policy_loss", []),   "Policy Loss",  "Loss",     "#e09a5c")
plot(ax_value,   data.get("value_loss", []),    "Value Loss",   "Loss",     "#5c8fe0")
plot(ax_winrate, data.get("against_random_player", []),
                                                "Win Rate vs Random", "Win Rate", "#5cc45c",
     x_values=[i * 5 for i in range(1, len(data.get("against_random_player", [])) + 1)])

# shade the "good" zone for value loss (clearly learning if well below 1.0)
if data.get("value_loss"):
    ax_value.axhline(1.0, color="grey", linestyle="--", linewidth=1,
                     label="random baseline (≈1.0)")
    ax_value.legend(fontsize=8)

# win-rate y-axis fixed to [0, 1]
ax_winrate.set_ylim(0, 1.05)
ax_winrate.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))

plt.savefig("training_logs_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved training_logs_plot.png")