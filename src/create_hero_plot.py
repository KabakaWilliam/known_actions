import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ---------------------------
# Data
# ---------------------------
families = [
    "Seed-2", "GPT-5", "Claude 4", "Gemini-3", "Gemini-3-Flash",
    "Gemma-4", "GLM-4.6V", "Qwen3-VL", "Qwen3.5", "UI-TARS-1.5"
]

datasets = {
    "2WikiMultihopQA":     [95.3, 90.2, 64.4, 68.2, 77.2, 86.5, 68.4, 90.7, 75.4, 93.0],
    "FRAMES":    [95.9, 80.8, 65.8, 72.5, 52.4, 77.6, 75.9, 94.7, 76.6, 91.9],
    "WebShop":   [93.9, 74.4, 67.2, 75.2, 71.8, 76.7, 67.1, 91.3, 79.6, 90.5],
    "DeepShop":  [87.2, 69.1, 73.5, 60.9, 68.5, 69.9, 71.0, 87.3, 76.4, 79.4],
}

# First 5 = proprietary, last 5 = open-source
prop_color = "#e9b59e"
open_color = "#98abd0"
edge_color = "#7a7a7a"

bar_colors = [prop_color] * 5 + [open_color] * 5

# ---------------------------
# Figure setup
# ---------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
axes = axes.flatten()

panel_labels = ["A", "B", "C", "D"]
x = np.arange(len(families))

for ax, (panel_label, (dataset_name, values)) in zip(axes, zip(panel_labels, datasets.items())):
    bars = ax.bar(
        x,
        values,
        width=0.48,
        color=bar_colors,
        edgecolor=edge_color,
        linewidth=0.9,
        zorder=3,
    )

    # Panel label
    ax.text(
        0.01, 1.05, panel_label,
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

    # Panel title
    ax.set_title(dataset_name, fontsize=24, pad=14)

    # Axes and grid
    ax.set_ylim(0, 101)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.9, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # X labels
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=40, ha="right", fontsize=11)

    # Tick styling
    ax.tick_params(axis="y", labelsize=11)

    # Value labels
    for rect, val in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + 1.2,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

# Ensure right column also shows y tick labels
for ax in [axes[1], axes[3]]:
    ax.tick_params(axis="y", labelleft=True)

# Shared y-axis label
fig.supylabel("Macro F1 (%)", fontsize=20, x=0.03)

# Legend
legend_handles = [
    Patch(facecolor=prop_color, edgecolor=edge_color, label="Proprietary families"),
    Patch(facecolor=open_color, edgecolor=edge_color, label="Open-source families"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=2,
    frameon=False,
    fontsize=15,
    bbox_to_anchor=(0.5, -0.01),
)

# Layout
fig.tight_layout(rect=[0.04, 0.07, 1, 1])

# Save
plt.savefig("figures/family_identifiability_barplots.png", dpi=300, bbox_inches="tight")
# plt.savefig("figures/family_identifiability_barplots.pdf", bbox_inches="tight")
plt.show()