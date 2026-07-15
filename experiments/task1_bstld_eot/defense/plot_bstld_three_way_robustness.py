from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV_PATH = Path(
    "experiments/task1_bstld_eot/"
    "defense/results/"
    "bstld_three_way_robustness_comparison.csv"
)

OUTPUT_PATH = Path(
    "experiments/task1_bstld_eot/"
    "figures/"
    "bstld_three_way_robustness_comparison.png"
)


def main():
    data = pd.read_csv(CSV_PATH)

    print("Columns:")
    print(data.columns.tolist())

    print()
    print("Three-way robustness comparison:")
    print(data.to_string(index=False))

    labels = [
        "Classification\naccuracy",
        "Targeted\nmisclassification",
        "Vanishing",
    ]

    original = [
        data.loc[
            data["evaluation"] == "Original attacked model",
            "classification_accuracy_percent",
        ].iloc[0],
        data.loc[
            data["evaluation"] == "Original attacked model",
            "tmr_percent",
        ].iloc[0],
        data.loc[
            data["evaluation"] == "Original attacked model",
            "vanishing_percent",
        ].iloc[0],
    ]

    fixed = [
        data.loc[
            data["evaluation"] == "Defended model fixed attack",
            "classification_accuracy_percent",
        ].iloc[0],
        data.loc[
            data["evaluation"] == "Defended model fixed attack",
            "tmr_percent",
        ].iloc[0],
        data.loc[
            data["evaluation"] == "Defended model fixed attack",
            "vanishing_percent",
        ].iloc[0],
    ]

    adaptive = [
        data.loc[
            data["evaluation"] == "Defended model adaptive attack",
            "classification_accuracy_percent",
        ].iloc[0],
        data.loc[
            data["evaluation"] == "Defended model adaptive attack",
            "tmr_percent",
        ].iloc[0],
        data.loc[
            data["evaluation"] == "Defended model adaptive attack",
            "vanishing_percent",
        ].iloc[0],
    ]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(
        figsize=(11, 7),
    )

    bars_original = ax.bar(
        x - width,
        original,
        width,
        label="Original model + attack",
    )

    bars_fixed = ax.bar(
        x,
        fixed,
        width,
        label="Defense + fixed attack",
    )

    bars_adaptive = ax.bar(
        x + width,
        adaptive,
        width,
        label="Defense + adaptive attack",
    )

    ax.set_ylabel("Rate (%)")
    ax.set_title(
        "BSTLD Targeted EOT Attack and Defense Robustness"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.legend()

    for bars in [
        bars_original,
        bars_fixed,
        bars_adaptive,
    ]:
        ax.bar_label(
            bars,
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
