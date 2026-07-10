import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

src_img_dir = Path("/home/chi-chi/REU/LISAyoloFormat/train/images")
src_label_dir = Path("/home/chi-chi/REU/LISAyoloFormat/train/labels")

out_img_dir = Path("/home/chi-chi/REU/LISAyoloFormat_defense/train/images")
out_label_dir = Path("/home/chi-chi/REU/LISAyoloFormat_defense/train/labels")

# Use a simple synthetic patch for defense augmentation.
# This does not need to be the exact learned patch; it teaches robustness to occluding patches.
patch_size_ratio = 0.12
num_augmented = 1000
target_class = 0  # Red

image_files = list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.png"))
random.shuffle(image_files)

created = 0

for img_path in image_files:
    if created >= num_augmented:
        break

    label_path = src_label_dir / f"{img_path.stem}.txt"
    if not label_path.exists():
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()

    red_boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5 and int(float(parts[0])) == target_class:
            red_boxes.append(parts)

    if not red_boxes:
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    h, w = img.shape[:2]

    # Pick one red box
    parts = random.choice(red_boxes)
    _, x_c, y_c, bw, bh = map(float, parts[:5])

    x_c *= w
    y_c *= h
    bw *= w
    bh *= h

    patch_size = int(max(bw, bh) * 2.0)
    patch_size = max(12, patch_size)

    x1 = int(x_c - patch_size / 2)
    y1 = int(y_c + bh / 2)

    x2 = x1 + patch_size
    y2 = y1 + patch_size

    if x1 < 0 or y1 < 0 or x2 >= w or y2 >= h:
        continue

    # Random high-contrast patch
    patch = np.random.randint(0, 256, (patch_size, patch_size, 3), dtype=np.uint8)

    img[y1:y2, x1:x2] = patch

    new_img_name = f"{img_path.stem}_defense_patch{img_path.suffix}"
    new_label_name = f"{img_path.stem}_defense_patch.txt"

    cv2.imwrite(str(out_img_dir / new_img_name), img)
    shutil.copy(label_path, out_label_dir / new_label_name)

    created += 1

print(f"Created {created} defense-augmented images.")
