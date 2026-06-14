from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. Data
# ============================================================
data = [
    # Method, Family, P, R, mAP50, mAP50:95, Params, FPS
    ["YOLOv10-m", "Best YOLO-style", 56.00, 41.43, 44.00, 26.81, 15.30, 82.94],
    ["ATSS", "Best CNN-based", 24.09, 48.97, 25.50, 15.00, 32.13, 36.09],
    ["Deformable DETR", "Best DETR-based", 46.52, 39.27, 43.20, 26.47, 40.00, 35.07],
    ["RT-DETR-R18", "RT-DETR baseline", 58.91, 43.58, 42.74, 25.27, 19.88, 96.59],
    ["Ours", "Proposed", 56.64, 49.29, 46.43, 26.89, 20.25, 3.89],
]

columns = ["Method", "Family", "P", "R", "mAP50", "mAP50_95", "Params", "FPS"]
df = pd.DataFrame(data, columns=columns)


# ============================================================
# 2. Output
# ============================================================
current_dir = Path(__file__).resolve().parent
out_dir = current_dir / "figures"
out_dir.mkdir(exist_ok=True)

out_png = out_dir / "visdrone_best_family_ranking_flow.png"
out_pdf = out_dir / "visdrone_best_family_ranking_flow.pdf"


# ============================================================
# 3. Paper style
# ============================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# ============================================================
# 4. Compute ranks
# ============================================================
metrics = [
    ("R", "Recall"),
    ("mAP50", r"mAP$_{50}$"),
    ("mAP50_95", r"mAP$_{50:95}$"),
]

for metric, _ in metrics:
    df[f"{metric}_rank"] = df[metric].rank(ascending=False, method="min").astype(int)

x = np.arange(len(metrics))


# ============================================================
# 5. Line styles
# ============================================================
styles = {
    "YOLOv10-m": {
        "color": "#4C78A8",
        "marker": "o",
        "linewidth": 2.2,
        "markersize": 7,
    },
    "ATSS": {
        "color": "#F58518",
        "marker": "s",
        "linewidth": 2.2,
        "markersize": 7,
    },
    "Deformable DETR": {
        "color": "#54A24B",
        "marker": "^",
        "linewidth": 2.2,
        "markersize": 7,
    },
    "RT-DETR-R18": {
        "color": "#7F7F7F",
        "marker": "D",
        "linewidth": 2.2,
        "markersize": 7,
    },
    "Ours": {
        "color": "#C62828",
        "marker": "*",
        "linewidth": 3.2,
        "markersize": 14,
    },
}


# ============================================================
# 6. Draw figure
# ============================================================
fig, ax = plt.subplots(figsize=(7.6, 4.9), dpi=300)

# subtle horizontal rank bands
max_rank = len(df)
for rank in range(1, max_rank + 1):
    if rank % 2 == 0:
        ax.axhspan(rank - 0.5, rank + 0.5, color="#F7F7F7", zorder=0)

# draw each selected representative model
for _, row in df.iterrows():
    method = row["Method"]
    style = styles[method]

    ranks = [row[f"{metric}_rank"] for metric, _ in metrics]
    values = [row[metric] for metric, _ in metrics]

    ax.plot(
        x,
        ranks,
        color=style["color"],
        linewidth=style["linewidth"],
        marker=style["marker"],
        markersize=style["markersize"],
        label=method,
        zorder=5 if method == "Ours" else 3,
    )

    # value labels only
    for i, (rank, value) in enumerate(zip(ranks, values)):
        if method == "Ours":
            dy = -0.22
            fontweight = "bold"
            fontsize = 8.5
        else:
            dy = 0.25
            fontweight = "normal"
            fontsize = 8.0

        ax.text(
            i + 0.04,
            rank + dy,
            f"{value:.2f}",
            fontsize=fontsize,
            color=style["color"],
            fontweight=fontweight,
            ha="left",
            va="center",
            zorder=8,
        )

    # model name at the left side
    ax.text(
        -0.12,
        ranks[0],
        method,
        fontsize=8.8 if method != "CrossSight-RTDETR" else 9.5,
        color=style["color"],
        fontweight="bold" if method == "CrossSight-RTDETR" else "normal",
        ha="right",
        va="center",
        zorder=8,
    )


# ============================================================
# 7. Axes
# ============================================================
ax.set_xticks(x)
ax.set_xticklabels([name for _, name in metrics], fontsize=11, fontweight="bold")

ax.set_ylabel("Rank among selected representative detectors", fontsize=10.5, fontweight="bold")
ax.set_ylim(max_rank + 0.55, 0.45)
ax.set_yticks(range(1, max_rank + 1))
ax.set_yticklabels([str(i) for i in range(1, max_rank + 1)], fontsize=9)

ax.set_xlim(-0.55, len(metrics) - 1 + 0.45)

ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
ax.grid(axis="x", linestyle=":", linewidth=0.65, alpha=0.35)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_title(
    "Ranking Flow of the Best Representative Models on VisDrone2019",
    fontsize=12.5,
    fontweight="bold",
    pad=10,
)

ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.25),
    ncol=3,
    frameon=False,
    fontsize=8.5,
)


# ============================================================
# 8. Save
# ============================================================
plt.tight_layout()
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()

print("Saved figures:")
print(out_png)
print(out_pdf)