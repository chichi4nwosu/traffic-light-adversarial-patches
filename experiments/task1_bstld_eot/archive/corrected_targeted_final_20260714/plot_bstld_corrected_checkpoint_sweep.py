from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CSV_PATH = Path(
    "experiments/task1_bstld_eot/"
    "evaluation/bstld_eot_corrected_checkpoint_sweep.csv"
)

OUTPUT_PATH = Path(
    "experiments/task1_bstld_eot/"
    "figures/bstld_eot_corrected_checkpoint_trajectory.png"
)


def main():
    data = pd.read_csv(CSV_PATH)

    print("Columns:")
    print(data.columns.tolist())

    print()
    print("Checkpoint data:")
    print(data.to_string(index=False))

    epochs = data["epoch"]

    accuracy = data[
        "classification_accuracy_percent"
    ]

    tmr = data["tmr_percent"]

    vanishing = data["vanishing_percent"]

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        accuracy,
        marker="o",
        label="Classification Accuracy",
    )

    plt.plot(
        epochs,
        tmr,
        marker="o",
        label="Targeted Misclassification Rate",
    )

    plt.plot(
        epochs,
        vanishing,
        marker="o",
        label="Vanishing Rate",
    )

    plt.axvline(
        x=1,
        linestyle="--",
        label="Selected Targeted Checkpoint",
    )

    plt.xlabel("Training Epoch")
    plt.ylabel("Rate (%)")

    plt.title(
        "BSTLD EOT Attack Behavior Across Training"
    )

    plt.xticks(epochs)

    plt.ylim(0, 100)

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
