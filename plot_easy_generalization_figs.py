import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("paper_figs_easy_v3", exist_ok=True)

# Use editable TrueType fonts in PDF
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# Paper-standard dataset names
data = [
    ["TinyPerson",   "Baseline", 51.66, 41.92, 40.06, 13.50],
    ["TinyPerson",   "Ours",     60.52, 52.25, 53.40, 17.56],

    ["VisAlgae2023", "Baseline", 83.80, 72.91, 80.62, 58.88],
    ["VisAlgae2023", "Ours",     89.50, 87.90, 91.90, 68.50],

    ["GTSDB",        "Baseline", 99.45, 96.71, 99.09, 86.01],
    ["GTSDB",        "Ours",     97.20, 96.30, 99.10, 87.30],

    ["SIRST",        "Baseline", 52.65, 67.89, 63.41, 28.68],
    ["SIRST",        "Ours",     91.81, 92.60, 95.29, 48.08],

    ["UAVOD-10",     "Baseline", 40.31, 38.29, 35.81, 17.10],
    ["UAVOD-10",     "Ours",     49.29, 44.45, 45.96, 22.39],

    ["SARD 2",       "Baseline", 88.00, 71.92, 72.33, 48.55],
    ["SARD 2",       "Ours",     76.82, 74.56, 73.87, 48.71],
]

df = pd.DataFrame(
    data,
    columns=["Dataset", "Method", "Precision", "Recall", "mAP50", "mAP50_95"]
)

order = [
    "TinyPerson",
    "VisAlgae2023",
    "GTSDB",
    "SIRST",
    "UAVOD-10",
    "SARD 2"
]

base = df[df["Method"] == "Baseline"].set_index("Dataset").loc[order]
ours = df[df["Method"] == "Ours"].set_index("Dataset").loc[order]

df.to_csv("paper_figs_easy_v3/cross_domain_results.csv", index=False)


# ============================================================
# Figure 1: Baseline vs Ours grouped bar chart
# ============================================================

x = np.arange(len(order))
width = 0.36

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

axes[0].bar(
    x - width / 2,
    base["mAP50"],
    width,
    label="RT-DETR-R18",
    hatch="//",
    edgecolor="black",
    linewidth=0.8
)
axes[0].bar(
    x + width / 2,
    ours["mAP50"],
    width,
    label="CrossSight-RTDETR",
    hatch="\\\\",
    edgecolor="black",
    linewidth=0.8
)
axes[0].set_ylabel("mAP50 (%)")
axes[0].set_title("(a) mAP50")
axes[0].set_xticks(x)
axes[0].set_xticklabels(order, rotation=25, ha="right")
axes[0].set_ylim(0, 105)
axes[0].grid(axis="y", linestyle="--", alpha=0.35)

axes[1].bar(
    x - width / 2,
    base["mAP50_95"],
    width,
    label="RT-DETR-R18",
    hatch="//",
    edgecolor="black",
    linewidth=0.8
)
axes[1].bar(
    x + width / 2,
    ours["mAP50_95"],
    width,
    label="CrossSight-RTDETR",
    hatch="\\\\",
    edgecolor="black",
    linewidth=0.8
)
axes[1].set_ylabel("mAP50:95 (%)")
axes[1].set_title("(b) mAP50:95")
axes[1].set_xticks(x)
axes[1].set_xticklabels(order, rotation=25, ha="right")
axes[1].set_ylim(0, 95)
axes[1].grid(axis="y", linestyle="--", alpha=0.35)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("paper_figs_easy_v3/1_cross_domain_bar_map.pdf", bbox_inches="tight")
plt.savefig("paper_figs_easy_v3/1_cross_domain_bar_map.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# Figure 2: Slope / dumbbell chart for mAP50 and mAP50:95
# ============================================================

y = np.arange(len(order))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)


def draw_slope(ax, metric, xlabel, title, xlim):
    for i, dataset in enumerate(order):
        b = base.loc[dataset, metric]
        o = ours.loc[dataset, metric]

        ax.plot([b, o], [i, i], linewidth=2.0, alpha=0.9)
        ax.scatter(
            [b], [i],
            s=70,
            marker="o",
            edgecolor="black",
            linewidth=0.8,
            label="RT-DETR-R18" if i == 0 else None
        )
        ax.scatter(
            [o], [i],
            s=70,
            marker="s",
            edgecolor="black",
            linewidth=0.8,
            label="CrossSight-RTDETR" if i == 0 else None
        )

        # Avoid label overlap when two values are very close, e.g., GTSDB
        if abs(o - b) < 1.0:
            ax.text(b - 1.2, i + 0.13, f"{b:.2f}", ha="right", va="bottom", fontsize=8)
            ax.text(o + 1.2, i - 0.18, f"{o:.2f}", ha="left", va="top", fontsize=8)
        else:
            ax.text(b, i + 0.13, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
            ax.text(o, i - 0.18, f"{o:.2f}", ha="center", va="top", fontsize=8)

    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.grid(axis="x", linestyle="--", alpha=0.35)


draw_slope(
    axes[0],
    metric="mAP50",
    xlabel="mAP50 (%)",
    title="(a) mAP50",
    xlim=(30, 103)
)

draw_slope(
    axes[1],
    metric="mAP50_95",
    xlabel="mAP50:95 (%)",
    title="(b) mAP50:95",
    xlim=(10, 90)
)

axes[0].set_yticks(y)
axes[0].set_yticklabels(order)
axes[0].invert_yaxis()

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("paper_figs_easy_v3/2_cross_domain_slope_map50_map5095.pdf", bbox_inches="tight")
plt.savefig("paper_figs_easy_v3/2_cross_domain_slope_map50_map5095.png", dpi=300, bbox_inches="tight")
plt.close()

print("Done. Figures saved to paper_figs_easy_v3/")
print(df)