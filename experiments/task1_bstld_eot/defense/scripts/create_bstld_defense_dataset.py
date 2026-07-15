import random
import shutil
from pathlib import Path

import cv2
import numpy as np


RANDOM_SEED = 1

SRC_ROOT = Path(
    "/home/chi-chi/REU/Project_TL/datasets/BSTLD"
)

SRC_IMG_DIR = SRC_ROOT / "images" / "train"
SRC_LABEL_DIR = SRC_ROOT / "labels" / "train"

OUT_ROOT = Path(
    "/home/chi-chi/REU/BSTLD_defense"
)

OUT_IMG_DIR = OUT_ROOT / "train" / "images"
OUT_LABEL_DIR = OUT_ROOT / "train" / "labels"

SOURCE_CLASS = 0  # Red

NUM_AUGMENTED = 1000
PATCH_MULTIPLIER = 2.0
MIN_PATCH_SIZE = 12


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    OUT_IMG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_LABEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_files = sorted(
        list(SRC_IMG_DIR.glob("*.jpg"))
        + list(SRC_IMG_DIR.glob("*.jpeg"))
        + list(SRC_IMG_DIR.glob("*.png"))
    )

    print("Source images:", len(image_files))
    print("Target augmented images:", NUM_AUGMENTED)
    print("Source class:", SOURCE_CLASS)
    print("Patch multiplier:", PATCH_MULTIPLIER)

    created = 0
    skipped_no_label = 0
    skipped_no_red = 0
    skipped_bad_image = 0
    skipped_bounds = 0

    candidates = image_files.copy()
    random.shuffle(candidates)

    for img_path in candidates:
        if created >= NUM_AUGMENTED:
            break

        label_path = (
            SRC_LABEL_DIR
            / f"{img_path.stem}.txt"
        )

        if not label_path.exists():
            skipped_no_label += 1
            continue

        with open(label_path, "r") as file:
            lines = file.readlines()

        red_boxes = []

        for line in lines:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            class_id = int(float(parts[0]))

            if class_id == SOURCE_CLASS:
                red_boxes.append(parts)

        if not red_boxes:
            skipped_no_red += 1
            continue

        image = cv2.imread(str(img_path))

        if image is None:
            skipped_bad_image += 1
            continue

        image_height, image_width = image.shape[:2]

        parts = random.choice(red_boxes)

        _, x_center, y_center, bbox_width, bbox_height = map(
            float,
            parts[:5],
        )

        x_center *= image_width
        y_center *= image_height
        bbox_width *= image_width
        bbox_height *= image_height

        patch_size = int(
            max(
                bbox_width,
                bbox_height,
            )
            * PATCH_MULTIPLIER
        )

        patch_size = max(
            MIN_PATCH_SIZE,
            patch_size,
        )

        patch_start_x = int(
            x_center - patch_size / 2
        )

        patch_start_y = int(
            y_center + bbox_height / 2
        )

        patch_end_x = (
            patch_start_x + patch_size
        )

        patch_end_y = (
            patch_start_y + patch_size
        )

        if (
            patch_start_x < 0
            or patch_start_y < 0
            or patch_end_x >= image_width
            or patch_end_y >= image_height
        ):
            skipped_bounds += 1
            continue

        patch = np.random.randint(
            0,
            256,
            (
                patch_size,
                patch_size,
                3,
            ),
            dtype=np.uint8,
        )

        image[
            patch_start_y:patch_end_y,
            patch_start_x:patch_end_x,
        ] = patch

        new_img_name = (
            f"{img_path.stem}"
            f"_defense_patch"
            f"{img_path.suffix}"
        )

        new_label_name = (
            f"{img_path.stem}"
            "_defense_patch.txt"
        )

        cv2.imwrite(
            str(
                OUT_IMG_DIR
                / new_img_name
            ),
            image,
        )

        shutil.copy(
            label_path,
            OUT_LABEL_DIR
            / new_label_name,
        )

        created += 1

    print()
    print("=== BSTLD DEFENSE AUGMENTATION ===")
    print("Created:", created)
    print("Skipped no label:", skipped_no_label)
    print("Skipped no Red:", skipped_no_red)
    print("Skipped bad image:", skipped_bad_image)
    print("Skipped bounds:", skipped_bounds)

    if created != NUM_AUGMENTED:
        raise RuntimeError(
            f"Expected {NUM_AUGMENTED} augmented "
            f"images but created {created}."
        )

    print()
    print("PASS: BSTLD defense augmentation complete.")


if __name__ == "__main__":
    main()
