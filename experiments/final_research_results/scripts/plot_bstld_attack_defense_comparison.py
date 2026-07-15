from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV_PATH = Path(
    "experiments/task1_bstld_eot/"
    "defense/results/"
    "bstld_attack_defense_comparison.csv"
)

OUTPUT_PATH = Path(
    "experiments/task1_bstld_eot/"
    "figures/"
    "bstld_attack_defense_comparison.png"
)


def main():
    data = pd.read_csv(CSV_PATH)

    print("Columns:")
    print(data.columns.tolist())

    print()
    print("Attack-defense comparison:")
    print(data.to_string(index=False))

    labels = [
        "Classification\nAccuracy",
        "Targeted\nMisclassification Rate",
        "Vanishing\nRate",
    ]

    original = [
        data.loc[
            0,
            "classification_accuracy_percent",
        ],
        data.loc[
            0,
            "tmr_percent",
        ],
        data.loc[
            0,
            "vanishing_percent",
        ],
    ]

    defended = [
        data.loc[
            1,
            "classification_accuracy_percent",
        ],
        data.loc[
            1,
            "tmr_percent",
        ],
        data.loc[
            1,
            "vanishing_percent",
        ],
    ]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(
        figsize=(10, 6),
    )

    bars_original = ax.bar(
        x - width / 2,
        original,
        width,
        label="Original Model",
    )

    bars_defended = ax.bar(
        x + width / 2,
        defended,
        width,
        label="Defended Model",
    )

    ax.set_ylabel("Rate (%)")
    ax.set_title(
        "BSTLD EOT Patch Attack: "
        "Original vs. Defended Model"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.legend()

    ax.bar_label(
        bars_original,
        fmt="%.2f%%",
        padding=3,
    )

    ax.bar_label(
        bars_defended,
        fmt="%.2f%%",
        padding=3,
    )

    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
