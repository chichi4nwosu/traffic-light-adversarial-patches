from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV_PATH = Path(
    "experiments/task1_bstld_eot/"
    "defense/results/"
    "bstld_clean_model_comparison.csv"
)

OUTPUT_PATH = Path(
    "experiments/task1_bstld_eot/"
    "figures/"
    "bstld_clean_model_comparison.png"
)


def main():
    data = pd.read_csv(CSV_PATH)

    print("Columns:")
    print(data.columns.tolist())

    print()
    print("Clean model comparison:")
    print(data.to_string(index=False))

    labels = [
        "Precision",
        "Recall",
        "mAP50",
        "mAP50-95",
    ]

    original = [
        data.loc[0, "precision_percent"],
        data.loc[0, "recall_percent"],
        data.loc[0, "map50_percent"],
        data.loc[0, "map50_95_percent"],
    ]

    defended = [
        data.loc[1, "precision_percent"],
        data.loc[1, "recall_percent"],
        data.loc[1, "map50_percent"],
        data.loc[1, "map50_95_percent"],
    ]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(
        figsize=(10, 6),
    )

    original_bars = ax.bar(
        x - width / 2,
        original,
        width,
        label="Original Model",
    )

    defended_bars = ax.bar(
        x + width / 2,
        defended,
        width,
        label="Defended Model",
    )

    ax.set_ylabel("Metric (%)")
    ax.set_title(
        "BSTLD Clean Detection Performance: "
        "Original vs. Defended Model"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 80)
    ax.legend()

    ax.bar_label(
        original_bars,
        fmt="%.2f%%",
        padding=3,
    )

    ax.bar_label(
        defended_bars,
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
