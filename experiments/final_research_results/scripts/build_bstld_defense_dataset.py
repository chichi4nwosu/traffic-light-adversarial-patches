from pathlib import Path
import random
import shutil

import cv2
import numpy as np
import torch
import torch.nn.functional as F


SOURCE_ROOT = Path(
    "/home/chi-chi/REU/Project_TL/"
    "datasets/BSTLD"
)

OUTPUT_ROOT = Path(
    "/home/chi-chi/REU/Project_TL/"
    "datasets/BSTLD_defense"
)

PATCH_PATH = Path(
    "experiments/task1_bstld_eot/"
    "archive/corrected_targeted_final_20260714/"
    "bstld_eot_corrected_final_3.pt"
)

PATCH_WIDTH_MULTIPLIER = 2.5
TARGET_CLASS = 0
RANDOM_SEED = 1


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

    if (
        patch.shape[-1] == 3
        and patch.shape[0] != 3
    ):
        patch = patch.permute(2, 0, 1)

    return patch.clamp(0, 1)


def find_image(image_dir, stem):
    for extension in [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
    ]:
        path = image_dir / f"{stem}{extension}"

        if path.exists():
            return path

    return None


def read_red_boxes(
    label_path,
    image_width,
    image_height,
):
    boxes = []

    for line in label_path.read_text().splitlines():
        parts = line.strip().split()

        if len(parts) < 5:
            continue

        class_id = int(float(parts[0]))

        if class_id != TARGET_CLASS:
            continue

        x_center = float(parts[1]) * image_width
        y_center = float(parts[2]) * image_height
        bbox_width = float(parts[3]) * image_width
        bbox_height = float(parts[4]) * image_height

        x1 = x_center - bbox_width / 2
        y1 = y_center - bbox_height / 2
        x2 = x_center + bbox_width / 2
        y2 = y_center + bbox_height / 2

        boxes.append(
            (
                x1,
                y1,
                x2,
                y2,
            )
        )

    return boxes


def apply_patch(
    image,
    patch,
    red_boxes,
):
    patched_image = image.copy()

    image_height, image_width = image.shape[:2]
    patch_tensor = patch.unsqueeze(0)

    patches_placed = 0

    for gt_box in red_boxes:
        x1, y1, x2, y2 = gt_box

        bbox_width = x2 - x1
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
            size=(
                patch_size,
                patch_size,
            ),
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

        patch_array = cv2.cvtColor(
            patch_array,
            cv2.COLOR_RGB2BGR,
        )

        patched_image[
            patch_start_y:patch_end_y,
            patch_start_x:patch_end_x,
        ] = patch_array

        patches_placed += 1

    return patched_image, patches_placed


def copy_clean_split(split):
    source_images = (
        SOURCE_ROOT / "images" / split
    )

    source_labels = (
        SOURCE_ROOT / "labels" / split
    )

    output_images = (
        OUTPUT_ROOT / "images" / split
    )

    output_labels = (
        OUTPUT_ROOT / "labels" / split
    )

    output_images.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_labels.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_count = 0
    label_count = 0

    for image_path in source_images.iterdir():
        if not image_path.is_file():
            continue

        shutil.copy2(
            image_path,
            output_images / image_path.name,
        )

        image_count += 1

    for label_path in source_labels.glob("*.txt"):
        shutil.copy2(
            label_path,
            output_labels / label_path.name,
        )

        label_count += 1

    return image_count, label_count


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if OUTPUT_ROOT.exists():
        raise RuntimeError(
            f"Output dataset already exists: "
            f"{OUTPUT_ROOT}\n"
            "Refusing to overwrite it."
        )

    print("Source dataset:", SOURCE_ROOT)
    print("Output dataset:", OUTPUT_ROOT)
    print("Selected patch:", PATCH_PATH)

    patch = load_patch(PATCH_PATH)

    print("Patch shape:", patch.shape)

    train_images, train_labels = (
        copy_clean_split("train")
    )

    val_images, val_labels = (
        copy_clean_split("val")
    )

    print()
    print("Copied clean train images:", train_images)
    print("Copied clean train labels:", train_labels)
    print("Copied clean val images:", val_images)
    print("Copied clean val labels:", val_labels)

    source_image_dir = (
        SOURCE_ROOT / "images" / "train"
    )

    source_label_dir = (
        SOURCE_ROOT / "labels" / "train"
    )

    output_image_dir = (
        OUTPUT_ROOT / "images" / "train"
    )

    output_label_dir = (
        OUTPUT_ROOT / "labels" / "train"
    )

    label_paths = sorted(
        source_label_dir.glob("*.txt")
    )

    augmented_images = 0
    total_patches = 0
    red_images_seen = 0
    skipped_no_image = 0
    skipped_no_patch_fit = 0

    for label_path in label_paths:
        image_path = find_image(
            source_image_dir,
            label_path.stem,
        )

        if image_path is None:
            skipped_no_image += 1
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            skipped_no_image += 1
            continue

        image_height, image_width = image.shape[:2]

        red_boxes = read_red_boxes(
            label_path,
            image_width,
            image_height,
        )

        if not red_boxes:
            continue

        red_images_seen += 1

        patched_image, patches_placed = apply_patch(
            image,
            patch,
            red_boxes,
        )

        if patches_placed == 0:
            skipped_no_patch_fit += 1
            continue

        output_name = (
            f"{image_path.stem}"
            f"_eot_defense"
            f"{image_path.suffix}"
        )

        output_label_name = (
            f"{image_path.stem}"
            f"_eot_defense.txt"
        )

        success = cv2.imwrite(
            str(output_image_dir / output_name),
            patched_image,
        )

        if not success:
            raise RuntimeError(
                f"Failed to write: {output_name}"
            )

        shutil.copy2(
            label_path,
            output_label_dir / output_label_name,
        )

        augmented_images += 1
        total_patches += patches_placed

    data_yaml = OUTPUT_ROOT / "data.yaml"

    data_yaml.write_text(
        "path: "
        "/home/chi-chi/REU/Project_TL/"
        "datasets/BSTLD_defense\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        "  0: red\n"
        "  1: yellow\n"
        "  2: green\n"
        "  3: off\n"
    )

    print()
    print("=" * 60)
    print("BSTLD DEFENSE DATASET COMPLETE")
    print("=" * 60)
    print("Red training images seen:", red_images_seen)
    print("Augmented images created:", augmented_images)
    print("Total patches placed:", total_patches)
    print("Skipped missing images:", skipped_no_image)
    print(
        "Skipped because no patch fit:",
        skipped_no_patch_fit,
    )
    print("Dataset YAML:", data_yaml)


if __name__ == "__main__":
    main()
