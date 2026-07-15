from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from tqdm import tqdm


MODEL_PATH = (
    "runs/detect/bstld_eot_defense_50/"
    "weights/best.pt"
)

PATCH_PATH = (
    "experiments/task1_bstld_eot/"
    "archive/corrected_targeted_final_20260714/"
    "bstld_eot_corrected_final_3.pt"
)

DATASET_ROOT = Path(
    "/home/chi-chi/REU/Project_TL/datasets/BSTLD"
)

IMAGE_DIR = DATASET_ROOT / "images" / "val"
LABEL_DIR = DATASET_ROOT / "labels" / "val"

SOURCE_CLASS = 0
TARGET_CLASS = 2

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50
IMAGE_SIZE = 1280

PATCH_WIDTH_MULTIPLIER = 2.5

OUTPUT_PATH = (
    "experiments/task1_bstld_eot/"
    "evaluation/bstld_patch_metrics.txt"
)


def yolo_to_xyxy(label, image_width, image_height):
    cls_id = int(float(label[0]))

    x_center = float(label[1]) * image_width
    y_center = float(label[2]) * image_height
    box_width = float(label[3]) * image_width
    box_height = float(label[4]) * image_height

    x1 = x_center - box_width / 2
    y1 = y_center - box_height / 2
    x2 = x_center + box_width / 2
    y2 = y_center + box_height / 2

    return cls_id, torch.tensor(
        [x1, y1, x2, y2],
        dtype=torch.float32,
    )


def calculate_iou(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)
    intersection = intersection_width * intersection_height

    area_a = max(
        0.0,
        float(box_a[2] - box_a[0]),
    ) * max(
        0.0,
        float(box_a[3] - box_a[1]),
    )

    area_b = max(
        0.0,
        float(box_b[2] - box_b[0]),
    ) * max(
        0.0,
        float(box_b[3] - box_b[1]),
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def find_image(label_path):
    for extension in [".jpg", ".jpeg", ".png", ".bmp"]:
        image_path = IMAGE_DIR / f"{label_path.stem}{extension}"

        if image_path.exists():
            return image_path

    return None


def load_patch(path):
    patch = torch.load(
        path,
        map_location="cpu",
    )

    if isinstance(patch, dict):
        for key in [
            "patch",
            "delta",
            "adversarial_patch",
        ]:
            if key in patch:
                patch = patch[key]
                break

    if not torch.is_tensor(patch):
        patch = torch.tensor(patch)

    patch = patch.float()

    if patch.ndim == 4:
        patch = patch[0]

    if patch.shape[-1] == 3 and patch.shape[0] != 3:
        patch = patch.permute(2, 0, 1)

    return patch.clamp(0, 1)


def apply_patch(
    image,
    patch,
    gt_red_boxes,
):
    patched_image = image.copy()

    image_height, image_width = image.shape[:2]

    patch_tensor = patch.unsqueeze(0)

    patches_placed = 0
    patched_gt_boxes = []

    for gt_box in gt_red_boxes:
        x1, y1, x2, y2 = [
            float(value)
            for value in gt_box
        ]

        bbox_width = x2 - x1
        bbox_height = y2 - y1

        x_center = (x1 + x2) / 2

        patch_size = int(
            bbox_width
            * PATCH_WIDTH_MULTIPLIER
        )

        if patch_size <= 0:
            continue

        patch_start_x = int(
            x_center - patch_size / 2
        )

        patch_start_y = int(y2)

        patch_end_x = (
            patch_start_x + patch_size
        )

        patch_end_y = (
            patch_start_y + patch_size
        )

        if (
            patch_start_x < 0
            or patch_start_y < 0
            or patch_end_x > image_width
            or patch_end_y > image_height
        ):
            continue

        resized_patch = F.interpolate(
            patch_tensor,
            size=(patch_size, patch_size),
            mode="bilinear",
            align_corners=False,
        )[0]

        patch_array = (
            resized_patch
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )

        patch_array = (
            patch_array * 255.0
        ).clip(
            0,
            255,
        ).astype(np.uint8)

        # Training tensor is RGB channel order.
        # OpenCV images are BGR.
        patch_array = cv2.cvtColor(
            patch_array,
            cv2.COLOR_RGB2BGR,
        )

        patched_image[
            patch_start_y:patch_end_y,
            patch_start_x:patch_end_x,
        ] = patch_array

        patches_placed += 1
        patched_gt_boxes.append(gt_box)

    return (
        patched_image,
        patches_placed,
        patched_gt_boxes,
    )


def main():
    device = (
        0
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)
    print("Loading model:", MODEL_PATH)
    print("Loading patch:", PATCH_PATH)

    model = YOLO(MODEL_PATH)
    patch = load_patch(PATCH_PATH)

    print("Classes:", model.names)
    print("Patch shape:", patch.shape)

    label_paths = sorted(
        LABEL_DIR.glob("*.txt")
    )

    total_gt_red = 0
    detected_matched = 0
    correct_color = 0
    misclassified = 0
    targeted_green = 0
    vanished = 0

    total_patches_placed = 0
    skipped_patch_boxes = 0

    for label_path in tqdm(
        label_paths,
        desc="Evaluating BSTLD patch",
    ):
        image_path = find_image(label_path)

        if image_path is None:
            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        image_height, image_width = (
            image.shape[:2]
        )

        gt_red_boxes = []

        with open(label_path, "r") as file:
            for line in file:
                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                cls_id, gt_box = (
                    yolo_to_xyxy(
                        parts,
                        image_width,
                        image_height,
                    )
                )

                if cls_id == SOURCE_CLASS:
                    gt_red_boxes.append(gt_box)

        if not gt_red_boxes:
            continue

        (
            patched_image,
            patches_placed,
            patched_gt_boxes,
        ) = apply_patch(
            image,
            patch,
            gt_red_boxes,
        )

        total_patches_placed += patches_placed
        skipped_patch_boxes += (
            len(gt_red_boxes)
            - len(patched_gt_boxes)
        )

        if patches_placed == 0:
            continue

        result = model.predict(
            source=patched_image,
            imgsz=IMAGE_SIZE,
            conf=CONF_THRESHOLD,
            verbose=False,
            device=device,
        )[0]

        predicted_boxes = []

        if result.boxes is not None:
            for box in result.boxes:
                xyxy = (
                    box.xyxy[0]
                    .detach()
                    .cpu()
                )

                predicted_class = int(
                    box.cls[0]
                    .detach()
                    .cpu()
                    .item()
                )

                confidence = float(
                    box.conf[0]
                    .detach()
                    .cpu()
                    .item()
                )

                predicted_boxes.append(
                    (
                        xyxy,
                        predicted_class,
                        confidence,
                    )
                )

        for gt_box in patched_gt_boxes:
            total_gt_red += 1

            best_match = None
            best_iou = 0.0

            for (
                predicted_box,
                predicted_class,
                confidence,
            ) in predicted_boxes:
                iou = calculate_iou(
                    gt_box,
                    predicted_box,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_match = (
                        predicted_class,
                        confidence,
                    )

            if (
                best_match is None
                or best_iou < IOU_THRESHOLD
            ):
                vanished += 1
                continue

            detected_matched += 1

            predicted_class = best_match[0]

            if predicted_class == SOURCE_CLASS:
                correct_color += 1
            else:
                misclassified += 1

                if predicted_class == TARGET_CLASS:
                    targeted_green += 1

    classification_accuracy = (
        correct_color / detected_matched
        if detected_matched
        else 0.0
    )

    targeted_misclassification_rate = (
        targeted_green / detected_matched
        if detected_matched
        else 0.0
    )

    vanishing_rate = (
        vanished / total_gt_red
        if total_gt_red
        else 0.0
    )

    output = f"""
=== BSTLD EOT Patch Evaluation (IoU-matched) ===
Model: {MODEL_PATH}
Patch: {PATCH_PATH}

Total GT red TLs evaluated: {total_gt_red}
Total patches placed: {total_patches_placed}
Skipped patch boxes/images: {skipped_patch_boxes}

Detected (matched): {detected_matched}
Correct color: {correct_color}
Misclassified: {misclassified}
Targeted Green: {targeted_green}
Vanished: {vanished}

Classification accuracy: {classification_accuracy:.6f}
Classification accuracy (%): {classification_accuracy * 100:.2f}%

Targeted misclassification rate: {targeted_misclassification_rate:.6f}
Targeted misclassification rate (%): {targeted_misclassification_rate * 100:.2f}%

Vanishing rate: {vanishing_rate:.6f}
Vanishing rate (%): {vanishing_rate * 100:.2f}%
"""

    print(output)

    Path(OUTPUT_PATH).write_text(output)

    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
