import csv
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(
    "/home/chi-chi/REU/"
    "attacks-on-traffic-light-detection"
)

EVALUATOR = (
    REPO_ROOT
    / "experiments/task1_bstld_eot/"
    "evaluation/evaluate_bstld_final_patch.py"
)

PATCH_DIR = (
    REPO_ROOT
    / "experiments/task1_bstld_eot/"
    "patches/corrected_final"
)

WORK_EVALUATOR = (
    REPO_ROOT
    / "experiments/task1_bstld_eot/"
    "evaluation/_checkpoint_eval_temp.py"
)

CSV_PATH = (
    REPO_ROOT
    / "experiments/task1_bstld_eot/"
    "evaluation/bstld_eot_corrected_checkpoint_sweep.csv"
)


def extract(pattern, output):
    match = re.search(pattern, output)

    if match is None:
        raise RuntimeError(
            f"Could not extract metric: {pattern}"
        )

    return float(match.group(1))


def main():
    base_text = EVALUATOR.read_text()

    start = base_text.index("PATCH_PATH = (")
    end = base_text.index("\n\n", start)

    rows = []

    for epoch in range(1, 6):
        patch_path = (
            "experiments/task1_bstld_eot/"
            "patches/corrected_final/"
            f"bstld_eot_corrected_final_{epoch}.pt"
        )

        patch_file = REPO_ROOT / patch_path

        if not patch_file.exists():
            raise FileNotFoundError(patch_file)

        replacement = (
            f'PATCH_PATH = "{patch_path}"'
        )

        eval_text = (
            base_text[:start]
            + replacement
            + base_text[end:]
        )

        WORK_EVALUATOR.write_text(eval_text)

        print()
        print("=" * 70)
        print(f"EVALUATING EPOCH {epoch}")
        print("=" * 70)

        process = subprocess.run(
            [
                sys.executable,
                str(WORK_EVALUATOR),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        print(process.stdout)

        if process.returncode != 0:
            print(process.stderr)
            raise RuntimeError(
                f"Epoch {epoch} evaluation failed."
            )

        accuracy = extract(
            r"Classification accuracy \(%\): ([0-9.]+)%",
            process.stdout,
        )

        tmr = extract(
            r"Targeted misclassification rate \(%\): ([0-9.]+)%",
            process.stdout,
        )

        vanishing = extract(
            r"Vanishing rate \(%\): ([0-9.]+)%",
            process.stdout,
        )

        rows.append(
            {
                "epoch": epoch,
                "classification_accuracy_percent": accuracy,
                "tmr_percent": tmr,
                "vanishing_percent": vanishing,
            }
        )

        print(
            f"SUMMARY EPOCH {epoch}: "
            f"Accuracy={accuracy:.2f}% | "
            f"TMR={tmr:.2f}% | "
            f"Vanishing={vanishing:.2f}%"
        )

    with open(
        CSV_PATH,
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "classification_accuracy_percent",
                "tmr_percent",
                "vanishing_percent",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 70)
    print("BSTLD EOT CHECKPOINT SWEEP")
    print("=" * 70)

    print(
        f"{'Epoch':<8}"
        f"{'Accuracy':>12}"
        f"{'TMR':>12}"
        f"{'Vanishing':>14}"
    )

    for row in rows:
        print(
            f"{row['epoch']:<8}"
            f"{row['classification_accuracy_percent']:>11.2f}%"
            f"{row['tmr_percent']:>11.2f}%"
            f"{row['vanishing_percent']:>13.2f}%"
        )

    best_tmr = max(
        rows,
        key=lambda row: row["tmr_percent"],
    )

    print()
    print(
        "Highest observed TMR checkpoint:",
        f"epoch {best_tmr['epoch']}",
    )

    print(
        "TMR:",
        f"{best_tmr['tmr_percent']:.2f}%",
    )

    print(
        "Classification accuracy:",
        f"{best_tmr['classification_accuracy_percent']:.2f}%",
    )

    print(
        "Vanishing:",
        f"{best_tmr['vanishing_percent']:.2f}%",
    )

    print()
    print("Saved:", CSV_PATH)

    if WORK_EVALUATOR.exists():
        WORK_EVALUATOR.unlink()


if __name__ == "__main__":
    main()
