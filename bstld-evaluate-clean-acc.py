import os
import sys
import yaml
import torch
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from ultralytics import YOLO
from ultralytics.data import YOLODataset


# ---- Output folder (renamed) ----
OUTPUT_DIR = "outputs-clean-eval-v8m-iou"

# ---- Matching / detection settings ----
IOU_THRESH = 0.50      # a GT light counts as "detected" only if some prediction overlaps it at >= this IoU
CONF_THRESH = 0.25    # detection confidence threshold used at inference


def iou_xywh(box1, boxes):
    """
    IoU of a single box against many boxes, all in [cx, cy, w, h] format
    and all in the SAME coordinate space (here: normalized 0..1).

    box1:  tensor shape (4,)
    boxes: tensor shape (N, 4)
    returns: tensor shape (N,) of IoU values
    """
    # single box -> corners
    b1x1 = box1[0] - box1[2] / 2
    b1y1 = box1[1] - box1[3] / 2
    b1x2 = box1[0] + box1[2] / 2
    b1y2 = box1[1] + box1[3] / 2

    # many boxes -> corners
    bx1 = boxes[:, 0] - boxes[:, 2] / 2
    by1 = boxes[:, 1] - boxes[:, 3] / 2
    bx2 = boxes[:, 0] + boxes[:, 2] / 2
    by2 = boxes[:, 1] + boxes[:, 3] / 2

    inter_x1 = torch.max(b1x1, bx1)
    inter_y1 = torch.max(b1y1, by1)
    inter_x2 = torch.min(b1x2, bx2)
    inter_y2 = torch.min(b1y2, by2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = box1[2] * box1[3]
    area2 = boxes[:, 2] * boxes[:, 3]
    union = area1 + area2 - inter

    return inter / union.clamp(min=1e-9)


def clean_eval(params, model, loader, device):
    img_size = params["img_size"]
    class_to_check = params["class_to_replace"]

    total = 0          # number of GT red lights considered
    matched = 0        # GT red lights that were actually detected (IoU >= threshold)
    correct = 0        # detected AND classified as the right color
    misclassified = 0  # detected BUT wrong color
    vanished = 0       # not detected at all (no prediction overlaps the GT box)

    for batch in tqdm(loader, desc="Clean evaluating"):
        img = batch["img"].to(device).float() / 255.0
        cls = batch["cls"]
        bboxes = batch["bboxes"][0]   # normalized xywh, shape (num_gt, 4)

        if not torch.numel(bboxes):
            continue

        # Select the lowest GT box in the frame (largest normalized y-center),
        # matching the original script's choice of which GT light to score.
        y_coords = bboxes[:, 1]
        lowest_idx = torch.argmax(y_coords)
        gt_class = int(cls[:, lowest_idx, :].item())

        # Only score lights whose GT class is the one we care about (Red).
        if gt_class != class_to_check:
            continue

        total += 1

        # The GT box for this light (normalized [cx, cy, w, h]).
        gt_box = bboxes[lowest_idx].to(device).float()

        with torch.no_grad():
            result = model.predict(
                img, save=False, conf=CONF_THRESH, verbose=False
            )[0].boxes

        # No detections at all -> vanished.
        if result.cls.shape[0] == 0:
            vanished += 1
            continue

        # IMPORTANT: use xywhn (NORMALIZED) so it matches the normalized GT box.
        # Using result.xywh (pixels) against a normalized GT box would make every
        # IoU ~0 and everything would look vanished.
        preds_xywhn = result.xywhn.to(device).float()

        ious = iou_xywh(gt_box, preds_xywhn)
        best_iou, best_idx = torch.max(ious, dim=0)

        # No prediction overlaps the GT light enough -> detection failure (vanished).
        if best_iou.item() < IOU_THRESH:
            vanished += 1
            continue

        # The light was correctly detected. Now judge the color.
        matched += 1
        pred_class = int(result.cls[best_idx].item())

        if pred_class == class_to_check:
            correct += 1
        else:
            misclassified += 1

    # ---------- Report ----------
    print("\n=== Clean TL Classification Accuracy (IoU-matched) ===")
    print("Total GT red TLs:", total)
    print("Detected (matched):", matched)
    print("Correct color:", correct)
    print("Misclassified:", misclassified)
    print("Vanished:", vanished)

    if total > 0:
        # Detection axis
        print("\n-- Detection axis --")
        print("Detection recall:", matched / total)
        print("Vanishing rate:", vanished / total)

    if matched > 0:
        # Classification axis, CONDITIONED ON CORRECT DETECTION
        print("\n-- Classification axis (conditioned on correct detection) --")
        print("Conditional classification accuracy:", correct / matched)
        print("Conditional misclassification rate:", misclassified / matched)

    
    # ---------- Bar chart ----------
    labels = ["Correct", "Misclassified", "Vanished"]
    counts = [correct, misclassified, vanished]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "clean_eval_bar.png")
    print(f"\nSaving clean evaluation bar chart to {out_path}")

    plt.figure()
    plt.bar(labels, counts)
    for i, c in enumerate(counts):
        plt.text(i, c, str(c), ha="center", va="bottom")
    plt.title("Clean TL Classification Results (IoU-matched)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    if len(sys.argv) != 3:
        print("\nUsage:\n  python evaluate-clean-acc.py <config.yaml> <weights.pt>\n")
        sys.exit(1)

    config_file = sys.argv[1]
    weights_file = sys.argv[2]

    with open(config_file, "r") as f:
        params = yaml.safe_load(f)

    model = YOLO(weights_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    val_dataset = YOLODataset(
        params["val_img_folder_path"],
        imgsz=params["img_size"],
        augment=False,
        batch_size=1,
        data=dict(names=list(range(1000))),
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    clean_eval(params, model, val_loader, device)


if __name__ == "__main__":
    main()