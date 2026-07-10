import os
import glob
import re
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

out_dirs = [
    "/home/chi-chi/REU/Final_Project/evaluation/graphs",
    "/home/chi-chi/REU/Final_Project/paper/figures",
    "/home/chi-chi/REU/Final_Project/poster/figures",
]

for d in out_dirs:
    os.makedirs(d, exist_ok=True)

def save_all(name):
    for d in out_dirs:
        plt.savefig(os.path.join(d, name), dpi=300, bbox_inches="tight")
    plt.close()

# 1. Pipeline Diagram
steps = [
    "LISA Dataset",
    "Label Remapping\n0=Red, 1=Yellow, 2=Green",
    "Train YOLOv8m",
    "Clean Evaluation",
    "Train Adversarial Patch\nRed → Green",
    "Attack Baseline Model",
    "Patch-Augmented\nDefense Training",
    "Attack Defended Model",
    "Compare Results"
]

plt.figure(figsize=(8, 10))
for i, step in enumerate(steps):
    y = len(steps) - i
    plt.text(0.5, y, step, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="white"),
             fontsize=11)
    if i < len(steps) - 1:
        plt.arrow(0.5, y - 0.35, 0, -0.35, head_width=0.03, head_length=0.08, length_includes_head=True)

plt.xlim(0, 1)
plt.ylim(0, len(steps) + 1)
plt.axis("off")
plt.title("Project Pipeline: Attack and Defense Evaluation", fontsize=14)
save_all("project_pipeline_diagram.png")

# 2. Dataset Class Distribution
classes = ["Red", "Yellow", "Green"]
counts = [9296, 407, 7140]

plt.figure(figsize=(7, 5))
plt.bar(classes, counts)
plt.ylabel("Number of Labels")
plt.title("Remapped LISA Training Label Distribution")
for i, v in enumerate(counts):
    plt.text(i, v + 150, str(v), ha="center")
plt.grid(axis="y", alpha=0.3)
save_all("lisa_class_distribution.png")

# 3. Patch Evolution
patch_dir_candidates = [
    "/home/chi-chi/REU/Final_Project/attacks/remapped/outputs_train/patches",
    "/home/chi-chi/REU/attacks-on-traffic-light-detection/outputs_train/patches",
    "/home/chi-chi/REU/attacks-on-traffic-light-detection/outputs_train",
]

patch_files = []
for d in patch_dir_candidates:
    patch_files = glob.glob(os.path.join(d, "**", "patch_step_*.png"), recursive=True)
    if patch_files:
        break

def get_step(path):
    m = re.search(r"patch_step_(\d+)\.png", path)
    return int(m.group(1)) if m else -1

patch_files = sorted(patch_files, key=get_step)

if patch_files:
    selected_steps = [0, 200, 400, 600, 800, 996]
    selected = []
    for s in selected_steps:
        closest = min(patch_files, key=lambda p: abs(get_step(p) - s))
        selected.append(closest)

    plt.figure(figsize=(12, 3))
    for i, path in enumerate(selected):
        img = mpimg.imread(path)
        plt.subplot(1, len(selected), i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Step {get_step(path)}")
    plt.suptitle("Adversarial Patch Evolution During Training", fontsize=14)
    plt.tight_layout()
    save_all("patch_evolution.png")
else:
    print("No patch_step images found. Skipping patch evolution figure.")

print("Extra figures created.")
