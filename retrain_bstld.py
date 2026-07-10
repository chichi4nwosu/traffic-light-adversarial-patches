"""
BSTLD YOLOv8m Retraining Script
================================
Trains for 150 epochs with patience=30 so it stops early
if it genuinely plateaus. Previous run stopped at 50 epochs
and was clearly still improving.

Run with:
    python retrain_bstld.py

This will save the best model to:
    runs/detect/bstld_yolov8m_final/weights/best.pt
"""

from ultralytics import YOLO

def main():
    # Load pretrained YOLOv8m base (not your previous best.pt —
    # we're doing a clean retrain to avoid inheriting bad habits)
    model = YOLO("yolov8m.pt")

    results = model.train(
        data="/home/chi-chi/REU/Project_TL/datasets/BSTLD/data.yaml",
        epochs=150,
        patience=30,          # stops early if no improvement for 30 epochs
        imgsz=1280,           # same as your previous run
        batch=8,              # same as previous run
        device=0,
        workers=8,
        name="bstld_yolov8m_final",
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=False,
        close_mosaic=10,
        amp=True,
        plots=True,
        save=True,
        verbose=True,
        seed=0,
    )

    print("\n=== BSTLD Retraining Complete ===")
    print(f"Best model saved to: runs/detect/bstld_yolov8m_final/weights/best.pt")
    print(f"Final metrics:")
    print(f"  mAP50:    {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print(f"  Precision: {results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
    print(f"  Recall:    {results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")

if __name__ == "__main__":
    main()
