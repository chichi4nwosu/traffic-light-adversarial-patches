from pathlib import Path

from ultralytics import YOLO


BASELINE_MODEL = Path(
    "experiments/task1_bstld_eot/"
    "models/bstld_sumaiya_best.pt"
)

DATA_YAML = Path(
    "/home/chi-chi/REU/Project_TL/"
    "datasets/BSTLD_defense/data.yaml"
)

RUN_NAME = "bstld_eot_defense_50"


def main():
    print("=" * 70)
    print("BSTLD EOT DEFENSE TRAINING")
    print("=" * 70)

    print("Baseline model:", BASELINE_MODEL)
    print("Defense dataset:", DATA_YAML)
    print("Run name:", RUN_NAME)

    if not BASELINE_MODEL.exists():
        raise FileNotFoundError(
            BASELINE_MODEL
        )

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            DATA_YAML
        )

    model = YOLO(
        str(BASELINE_MODEL)
    )

    print()
    print("Loaded baseline classes:")
    print(model.names)

    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        patience=15,
        batch=8,
        imgsz=1280,
        device=0,
        workers=8,
        name=RUN_NAME,
        pretrained=True,
        seed=0,
        deterministic=True,
        val=True,
        plots=True,
    )

    print()
    print("=" * 70)
    print("BSTLD EOT DEFENSE TRAINING COMPLETE")
    print("=" * 70)

    print("Save directory:")
    print(results.save_dir)

    best_path = (
        Path(results.save_dir)
        / "weights"
        / "best.pt"
    )

    last_path = (
        Path(results.save_dir)
        / "weights"
        / "last.pt"
    )

    print("Best checkpoint:", best_path)
    print("Best exists:", best_path.exists())

    print("Last checkpoint:", last_path)
    print("Last exists:", last_path.exists())


if __name__ == "__main__":
    main()
