"""
Generate Final Paper Figures — Sumaiya's Style
================================================
Matches the figure style from CS691 Final Project Report.
Saves all figures to: final_results/paper_figures/

Run with:
    python generate_final_figures.py
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path("final_results/paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── numbers ────────────────────────────────────────────────
CLEAN_ACC     = 95.8   # baseline mAP50 %
ATTACK_ACC    = 23.9   # unaffected % under attack (proxy for accuracy)
DEFENSE_ACC   = 76.9   # unaffected % after defense

CLEAN_TMR     = 2.0    # baseline attack success on clean model
ATTACK_TMR    = 72.9   # attack success on baseline
DEFENSE_TMR   = 19.6   # attack success after defense

CLEAN_VAN     = 0.0
ATTACK_VAN    = 3.2
DEFENSE_VAN   = 3.5

# raw counts (n=536)
B_TARGET   = 391   # successful attacks baseline
B_OTHER    = 17    # vanishing baseline
B_UNAFFECT = 128   # unaffected baseline

D_TARGET   = 105
D_OTHER    = 19
D_UNAFFECT = 412

plt.rcParams.update({"font.size": 11})

RED    = "#e53935"
GREEN  = "#43a047"
GRAY   = "#9e9e9e"
BLUE   = "#1565C0"
ORANGE = "#E65100"


# ══════════════════════════════════════════════════════════
# Figure 1 — Before Defense (matches Sumaiya Fig 2a)
# ══════════════════════════════════════════════════════════
def fig_before_defense():
    categories = ["target_attack", "other_attack", "unattacked", "vanishing"]
    counts     = [B_TARGET, 0, B_UNAFFECT, B_OTHER]
    colors     = [RED, ORANGE, GREEN, GRAY]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(categories, counts, color=colors, width=0.5, edgecolor="black")
    ax.set_ylabel("Count")
    ax.set_title("EOT Evaluation Results (Before Defense)")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, counts):
        if val > 0:
            ax.annotate(str(val),
                        (bar.get_x() + bar.get_width()/2, val + 4),
                        ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "fig_before_defense.png", dpi=150)
    plt.close()
    print("✓ fig_before_defense.png")


# ══════════════════════════════════════════════════════════
# Figure 2 — After Defense (matches Sumaiya Fig 2b)
# ══════════════════════════════════════════════════════════
def fig_after_defense():
    categories = ["target_attack", "other_attack", "unattacked", "vanishing"]
    counts     = [D_TARGET, 0, D_UNAFFECT, D_OTHER]
    colors     = [RED, ORANGE, GREEN, GRAY]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(categories, counts, color=colors, width=0.5, edgecolor="black")
    ax.set_ylabel("Count")
    ax.set_title("EOT Evaluation Results (After Defense)")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, counts):
        if val > 0:
            ax.annotate(str(val),
                        (bar.get_x() + bar.get_width()/2, val + 4),
                        ha="center", va="bottom", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "fig_after_defense.png", dpi=150)
    plt.close()
    print("✓ fig_after_defense.png")


# ══════════════════════════════════════════════════════════
# Figure 3 — Side by side (matches Sumaiya Fig 2 combined)
# ══════════════════════════════════════════════════════════
def fig_combined():
    categories = ["target_attack", "other_attack", "unattacked", "vanishing"]
    before = [B_TARGET, 0, B_UNAFFECT, B_OTHER]
    after  = [D_TARGET, 0, D_UNAFFECT, D_OTHER]
    colors = [RED, ORANGE, GREEN, GRAY]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, counts, title in [
        (ax1, before, "EOT Evaluation Results (Before Defense)"),
        (ax2, after,  "EOT Evaluation Results (After Defense)"),
    ]:
        bars = ax.bar(categories, counts, color=colors, width=0.5, edgecolor="black")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(max(before), max(after)) * 1.15)
        for bar, val in zip(bars, counts):
            if val > 0:
                ax.annotate(str(val),
                            (bar.get_x() + bar.get_width()/2, val + 4),
                            ha="center", va="bottom", fontsize=11)

    plt.suptitle("Comparison of Adversarial Patch Effectiveness Before and After Defense",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT / "fig_combined_before_after.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ fig_combined_before_after.png")


# ══════════════════════════════════════════════════════════
# Figure 4 — Table I as image (matches Sumaiya Table I)
# ══════════════════════════════════════════════════════════
def fig_results_table():
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    columns = ["Metrics", "Clean Data", "EOT Patch Attack", "Defense (Robust)"]
    rows = [
        ["Classification Accuracy (%)",    f"{CLEAN_ACC:.2f}",  f"{ATTACK_ACC:.2f}",  f"{DEFENSE_ACC:.2f}"],
        ["Targeted Misclassification Rate (%)", f"{CLEAN_TMR:.2f}", f"{ATTACK_TMR:.2f}", f"{DEFENSE_TMR:.2f}"],
        ["Vanishing TL Rate (%)",           f"{CLEAN_VAN:.2f}",  f"{ATTACK_VAN:.2f}",  f"{DEFENSE_VAN:.2f}"],
    ]

    row_colors = [
        ["#ffffff"]*4,
        ["#f0f4ff"]*4,
        ["#ffffff"]*4,
    ]

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        cellColours=row_colors,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)

    for j in range(len(columns)):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        "TABLE I: Evaluation Metrics Under Clean, EOT Patch Attack, and Defense Conditions",
        fontsize=12, pad=20, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(OUT / "fig_results_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ fig_results_table.png")


# ══════════════════════════════════════════════════════════
# Figure 5 — Clean performance comparison
# ══════════════════════════════════════════════════════════
def fig_clean_performance():
    # per-class mAP50 baseline vs defense
    classes  = ["Red", "Yellow", "Green", "All"]
    baseline = [0.985, 0.943, 0.947, 0.958]
    defense  = [0.987, 0.963, 0.952, 0.965]

    x = np.arange(len(classes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, baseline, w, label="Baseline", color=BLUE)
    ax.bar(x + w/2, defense,  w, label="Defense",  color=ORANGE)
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("mAP50")
    ax.set_title("Clean Detection Performance: Baseline vs. Defense (LISA)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for rect in ax.patches:
        ax.annotate(f"{rect.get_height():.3f}",
                    (rect.get_x() + rect.get_width()/2, rect.get_height() + 0.001),
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "fig_clean_performance.png", dpi=150)
    plt.close()
    print("✓ fig_clean_performance.png")


# ══════════════════════════════════════════════════════════
# Figure 6 — Adversarial patch image
# ══════════════════════════════════════════════════════════
def fig_patch():
    patch_path = "outputs_train/final_patches/lisa_remapped_patch_FIXED_1.pt"
    if not os.path.exists(patch_path):
        print("✗ patch not found, skipping")
        return
    patch = torch.load(patch_path, map_location="cpu")
    patch_np = patch.permute(1, 2, 0).numpy().clip(0, 1)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.imshow(patch_np)
    ax.axis("off")
    ax.set_title("Optimized Adversarial Patch\n(Red → Green, LISA)", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "fig_adversarial_patch.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ fig_adversarial_patch.png")


# ══════════════════════════════════════════════════════════
# Figure 7 — Patch evolution
# ══════════════════════════════════════════════════════════
def fig_patch_evolution():
    epochs  = [1, 2, 3, 5, 10, 14]
    patches = []
    for e in epochs:
        p = f"outputs_train/final_patches/lisa_remapped_patch_FIXED_{e}.pt"
        if os.path.exists(p):
            pt = torch.load(p, map_location="cpu")
            patches.append((e, pt.permute(1,2,0).numpy().clip(0,1)))

    if not patches:
        print("✗ no patches found")
        return

    fig, axes = plt.subplots(1, len(patches), figsize=(3*len(patches), 3.5))
    if len(patches) == 1:
        axes = [axes]
    for ax, (epoch, p) in zip(axes, patches):
        ax.imshow(p)
        ax.axis("off")
        ax.set_title(f"Epoch {epoch}", fontsize=10)
    fig.suptitle("Adversarial Patch Evolution During Training", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "fig_patch_evolution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ fig_patch_evolution.png")


# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\nGenerating figures → {OUT}/\n")
    fig_before_defense()
    fig_after_defense()
    fig_combined()
    fig_results_table()
    fig_clean_performance()
    fig_patch()
    fig_patch_evolution()
    print(f"\nAll done. Files in {OUT}/")
    for f in sorted(OUT.glob("*.png")):
        print(f"  {f.name}")
