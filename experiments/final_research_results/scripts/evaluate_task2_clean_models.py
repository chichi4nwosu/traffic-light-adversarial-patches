from pathlib import Path
import csv

from ultralytics import YOLO


DATA_YAML = (
    "/home/chi-chi/REU/Project_TL/"
    "datasets/BSTLD/data.yaml"
)

OUTPUT_PATH = Path(
    "experiments/task2_bstld_transferability/"
    "results/task2_clean_model_comparison.csv"
)

MODELS = {
    "YOLOv8m": (
        "runs/detect/bstld_yolov8m_final3/"
        "weights/best.pt"
    ),
    "YOLOv8l": (
        "runs/detect/bstld_yolov8l/"
        "weights/best.pt"
    ),
}


def get_metric(results, key):
    value = results.results_dict.get(key)

    if value is None:
        raise RuntimeError(
            f"Metric not found: {key}\n"
            f"Available metrics: "
            f"{list(results.results_dict.keys())}"
        )

    return float(value)


def main():
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for model_name, model_path in MODELS.items():
        print()
        print("=" * 70)
        print(f"EVALUATING CLEAN MODEL: {model_name}")
        print("=" * 70)

        print("Model:", model_path)
        print("Data:", DATA_YAML)

        model = YOLO(model_path)

        results = model.val(
            data=DATA_YAML,
            imgsz=1280,
            batch=8,
            device=0,
            conf=0.25,
            iou=0.7,
            plots=False,
        )

        precision = get_metric(
            results,
            "metrics/precision(B)",
        )

        recall = get_metric(
            results,
            "metrics/recall(B)",
        )

        map50 = get_metric(
            results,
            "metrics/mAP50(B)",
        )

        map50_95 = get_metric(
            results,
            "metrics/mAP50-95(B)",
        )

        row = {
            "model": model_name,
            "model_path": model_path,
            "precision_percent": precision * 100,
            "recall_percent": recall * 100,
            "map50_percent": map50 * 100,
            "map50_95_percent": map50_95 * 100,
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
            f"mAP50-95: {map50_95 * 100:.2f}%"
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
                "model_path",
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
    print("TASK 2 CLEAN MODEL COMPARISON")
    print("=" * 70)

    for row in rows:
        print()
        print(row["model"])

        print(
            "Precision:",
            f"{row['precision_percent']:.2f}%",
        )

        print(
            "Recall:",
            f"{row['recall_percent']:.2f}%",
        )

        print(
            "mAP50:",
            f"{row['map50_percent']:.2f}%",
        )

        print(
            "mAP50-95:",
            f"{row['map50_95_percent']:.2f}%",
        )

    print()
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
