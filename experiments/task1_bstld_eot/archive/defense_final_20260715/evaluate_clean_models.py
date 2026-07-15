from pathlib import Path
import csv

from ultralytics import YOLO


DATA_YAML = (
    "/home/chi-chi/REU/Project_TL/"
    "datasets/BSTLD/data.yaml"
)

MODELS = {
    "Original model": (
        "experiments/task1_bstld_eot/"
        "models/bstld_sumaiya_best.pt"
    ),
    "Defended model": (
        "runs/detect/bstld_eot_defense_50/"
        "weights/best.pt"
    ),
}

OUTPUT_PATH = Path(
    "experiments/task1_bstld_eot/"
    "defense/results/"
    "bstld_clean_model_comparison.csv"
)


def main():
    rows = []

    for model_name, model_path in MODELS.items():
        print()
        print("=" * 70)
        print("EVALUATING:", model_name)
        print("MODEL:", model_path)
        print("=" * 70)

        model = YOLO(model_path)

        results = model.val(
            data=DATA_YAML,
            split="val",
            imgsz=1280,
            batch=8,
            device=0,
            conf=0.001,
            iou=0.7,
            plots=False,
            verbose=True,
        )

        metrics = results.results_dict

        precision = float(
            metrics["metrics/precision(B)"]
        )
        recall = float(
            metrics["metrics/recall(B)"]
        )
        map50 = float(
            metrics["metrics/mAP50(B)"]
        )
        map5095 = float(
            metrics["metrics/mAP50-95(B)"]
        )

        row = {
            "model": model_name,
            "precision_percent": precision * 100,
            "recall_percent": recall * 100,
            "map50_percent": map50 * 100,
            "map50_95_percent": map5095 * 100,
        }

        rows.append(row)

        print()
        print(
            f"{model_name} CLEAN SUMMARY"
        )
        print(
            f"Precision: {precision * 100:.2f}%"
        )
        print(
            f"Recall: {recall * 100:.2f}%"
        )
        print(
            f"mAP50: {map50 * 100:.2f}%"
        )
        print(
            f"mAP50-95: {map5095 * 100:.2f}%"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "precision_percent",
                "recall_percent",
                "map50_percent",
                "map50_95_percent",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 70)
    print("CLEAN MODEL COMPARISON")
    print("=" * 70)

    for row in rows:
        print(
            f"{row['model']}: "
            f"P={row['precision_percent']:.2f}% | "
            f"R={row['recall_percent']:.2f}% | "
            f"mAP50={row['map50_percent']:.2f}% | "
            f"mAP50-95={row['map50_95_percent']:.2f}%"
        )

    print()
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
