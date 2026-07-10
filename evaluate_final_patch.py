"""
Patch Evaluation Script
========================
Runs a saved adversarial patch (.pt file) against two models
(baseline and defense) and prints a clean comparison table.

Usage:
    python evaluate_final_patch.py

Results will be printed to terminal and saved to:
    final_evaluation_results.txt
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from ultralytics import YOLO
from ultralytics.data import YOLODataset
from tqdm import tqdm
from patch_utils.transforms import patch_resize

# ============================================================
# CONFIGURATION — edit these paths if needed
# ============================================================
PATCH_PATH      = "outputs_train/final_patches/lisa_remapped_patch_FIXED_1.pt"
BASELINE_MODEL  = "runs/detect/lisa_remapped_yolov8m/weights/best.pt"
DEFENSE_MODEL   = "runs/detect/lisa_defense_yolov8m_50/weights/best.pt"
VAL_IMAGES      = "/home/chi-chi/REU/LISAyoloFormat/test/images"

IMG_SIZE             = 640
CONF_THRES           = 0.25
CLASS_TO_REPLACE     = 0   # Red
TARGET_CLASS         = 2   # Green
PATCH_WIDTH_MULT     = 2
# ============================================================


def evaluate_model(model_path, patch, device, val_loader, model_label):
    print(f"\n{'='*55}")
    print(f"Evaluating: {model_label}")
    print(f"{'='*55}")

    model = YOLO(model_path)
    model.to(device)

    total   = 0
    success = 0
    vanish  = 0
    unaffected = 0

    for batch in tqdm(val_loader, desc=f"{model_label}", leave=False):
        val_img = batch["img"].to(device).float() / 255.0
        cls     = batch["cls"]
        bboxes  = batch["bboxes"][0]

        if not torch.numel(bboxes):
            continue

        # Only test images that have the attack class (Red)
        has_target = False
        for i in range(cls.shape[1]):
            if int(cls[0, i, 0].item()) == CLASS_TO_REPLACE:
                has_target = True
                break
        if not has_target:
            continue

        # Find the lowest red bbox (most visible)
        best_idx = None
        best_y   = -1
        for i in range(cls.shape[1]):
            if int(cls[0, i, 0].item()) == CLASS_TO_REPLACE:
                y = bboxes[i][1].item()
                if y > best_y:
                    best_y   = y
                    best_idx = i

        if best_idx is None:
            continue

        bbox        = bboxes[best_idx]
        x_center    = bbox[0].item() * IMG_SIZE
        y_center    = bbox[1].item() * IMG_SIZE
        bbox_width  = bbox[2].item() * IMG_SIZE
        bbox_height = bbox[3].item() * IMG_SIZE

        patch_size    = int(bbox_width * PATCH_WIDTH_MULT)
        patch_start_x = int(x_center - patch_size / 2)
        patch_start_y = int(y_center + bbox_height / 2)
        patch_end_x   = patch_start_x + patch_size
        patch_end_y   = patch_start_y + patch_size

        if (patch_end_x > IMG_SIZE or patch_end_y > IMG_SIZE
                or patch_start_x < 0 or patch_start_y < 0):
            continue

        # Place patch on image
        current_patch = patch_resize(patch, (patch_size, patch_size))
        adv_img = val_img.clone()
        for i in range(adv_img.shape[0]):
            adv_img[i, :, patch_start_y:patch_end_y, patch_start_x:patch_end_x] = current_patch

        # Run inference on patched image
        with torch.no_grad():
            result = model.predict(adv_img, save=False, conf=CONF_THRES, verbose=False)
            boxes  = result[0].boxes

            if boxes.cls.shape[0] == 0:
                vanish += 1
            else:
                y_coords = boxes.xywh[:, 1]
                top_idx  = torch.argmax(y_coords)
                pred_cls = int(boxes.cls[top_idx].item())
                if pred_cls == TARGET_CLASS:
                    success += 1
                else:
                    unaffected += 1

        total += 1

    if total == 0:
        print("WARNING: No valid test images found!")
        return {}

    success_rate    = success    / total * 100
    vanish_rate     = vanish     / total * 100
    unaffected_rate = unaffected / total * 100
    total_affected  = (success + vanish) / total * 100

    results = {
        "model":          model_label,
        "total":          total,
        "success":        success,
        "vanish":         vanish,
        "unaffected":     unaffected,
        "success_pct":    success_rate,
        "vanish_pct":     vanish_rate,
        "unaffected_pct": unaffected_rate,
        "total_affected": total_affected,
    }

    print(f"\n  Total images tested:    {total}")
    print(f"  Successful attacks:     {success} ({success_rate:.1f}%)")
    print(f"  Vanishing:              {vanish} ({vanish_rate:.1f}%)")
    print(f"  Unaffected:             {unaffected} ({unaffected_rate:.1f}%)")
    print(f"  Total affected:         {success + vanish} ({total_affected:.1f}%)")

    return results


def print_comparison(baseline, defense):
    print(f"\n{'='*55}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*55}")
    print(f"{'Metric':<25} {'Baseline':>12} {'Defense':>12} {'Change':>10}")
    print(f"{'-'*55}")

    metrics = [
        ("Attack Success (%)",    "success_pct"),
        ("Vanishing (%)",         "vanish_pct"),
        ("Total Affected (%)",    "total_affected"),
        ("Unaffected (%)",        "unaffected_pct"),
    ]

    lines = []
    for label, key in metrics:
        b = baseline[key]
        d = defense[key]
        delta = d - b
        arrow = "↑" if delta > 0 else "↓"
        sign  = "+" if delta > 0 else ""
        line  = f"{label:<25} {b:>11.1f}% {d:>11.1f}% {sign}{delta:.1f}% {arrow}"
        print(line)
        lines.append(line)

    print(f"{'='*55}")

    # Save to file
    with open("final_evaluation_results.txt", "w") as f:
        f.write(f"Patch: {PATCH_PATH}\n")
        f.write(f"Baseline model: {BASELINE_MODEL}\n")
        f.write(f"Defense model:  {DEFENSE_MODEL}\n\n")
        f.write(f"{'Metric':<25} {'Baseline':>12} {'Defense':>12} {'Change':>10}\n")
        f.write(f"{'-'*55}\n")
        for line in lines:
            f.write(line + "\n")

    print(f"\nResults saved to: final_evaluation_results.txt")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading patch: {PATCH_PATH}")

    patch = torch.load(PATCH_PATH, map_location=device)
    patch = patch.to(device)
    print(f"Patch shape: {patch.shape}")

    val_dataset = YOLODataset(
        VAL_IMAGES,
        imgsz=IMG_SIZE,
        augment=False,
        batch_size=1,
        data=dict(names=list(range(1000)))
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(f"Validation images: {len(val_dataset)}")

    baseline_results = evaluate_model(
        BASELINE_MODEL, patch, device, val_loader, "Baseline (no defense)"
    )
    defense_results = evaluate_model(
        DEFENSE_MODEL, patch, device, val_loader, "Defense model"
    )

    print_comparison(baseline_results, defense_results)


if __name__ == "__main__":
    main()
