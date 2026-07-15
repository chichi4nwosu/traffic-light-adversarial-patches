"""
FIXED Adversarial Patch Training Script for YOLOv8
====================================================
Key bug fixed: gradients now flow through delta correctly.

Original bug: TV loss and adv_img were computed on `current_patch`
(a detached transformed copy of delta), so delta received zero gradient
and the patch never learned anything — staying as random noise.

Fix: resize delta directly for placement in the image and compute
TV loss on delta, keeping the computation graph intact.

Run with:
    python train_patch_yolov8_fixed.py configs/lisa_remapped_attack_fast.yaml
"""

import cv2
from datetime import datetime
from tqdm import tqdm
import yaml
import platform
import wandb

import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.data import YOLODataset
from ultralytics.cfg import get_cfg

from patch_utils.losses import yolo_loss, total_variation_loss, v8AttackLoss
from patch_utils.transforms import patch_brightness, patch_pad, patch_rotate, patch_resize
from ultralytics.utils.loss import v8DetectionLoss
from patch_utils.general import load_classes, store_wandb_image_with_bboxes
from patch_utils.losses import yolo_loss, total_variation_loss, green_channel_penalty, similarity_loss
from patch_utils.transforms import patch_brightness, patch_pad, patch_rotate, patch_resize


def transform_patch_for_display(current_patch):
    """
    Transform patch for display/saving purposes only.
    Do NOT use the output of this for loss computation —
    it breaks the gradient graph.
    """
    brightness = np.random.uniform(0.4, 1.6)
    current_patch = patch_brightness(current_patch, brightness)
    pad_width = 5
    current_patch = patch_pad(current_patch, pad_width)
    mask = torch.ones_like(current_patch)
    anglex = np.random.uniform(-5, 5)
    angley = np.random.uniform(-5, 5)
    anglez = np.random.uniform(-10, 10)
    current_patch = patch_rotate(current_patch, anglex, angley, anglez)
    mask = patch_rotate(mask, anglex, angley, anglez)
    return current_patch, mask


def load_initial_patch(image_path, patch_size):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    patch = torch.from_numpy(image).permute(2, 0, 1)
    patch = patch_resize(patch, (patch_size, patch_size))
    return patch.squeeze(0)


def run_attack(params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = YOLO(params["yolo_weights_path"])
    model.to(device)
    classes = model.names
    img_size = params["img_size"]

    train_dataset = YOLODataset(
        params["train_img_folder_path"],
        imgsz=img_size,
        augment=False,
        batch_size=1,
        data=dict(names=list(range(1000)))
    )
    val_dataset = YOLODataset(
        params["val_img_folder_path"],
        imgsz=img_size,
        augment=False,
        batch_size=1,
        data=dict(names=list(range(1000)))
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    nb = len(train_loader)

    # ------------------------------------------------------------------ #
    # Initialise patch delta — this is the ONLY tensor that gets updated  #
    # ------------------------------------------------------------------ #
    delta = torch.empty(3, params["patch_size"], params["patch_size"]).uniform_(0, 1).to(device)
    delta.requires_grad = True

    optimizer = optim.Adam([delta], lr=params["learning_rate"], amsgrad=False)
    optimizer.zero_grad()

    model.model.train()
    model.model.args["imgsz"] = img_size
    model.model.args = get_cfg(model.model.args, None)
    v8loss = v8AttackLoss(model.model)

    step = 0
    saved_patch_i = 0

    for epoch in tqdm(range(params["num_epochs"]), desc="Training Epochs"):
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            img = batch["img"].to(device).float() / 255.0
            cls = batch["cls"]
            bbox_coords = batch["bboxes"][0]  # xywh

            if not torch.numel(bbox_coords):
                continue

            # Identify which bboxes to attack
            if params["only_lowest_bbox"]:
                y_coords = bbox_coords[:, 1]
                tl_indices_to_attack = [torch.argmax(y_coords).item()]
            else:
                tl_indices_to_attack = list(range(len(bbox_coords)))

            # Check if any bbox has the class we want to attack
            found_required_class = False
            for i in tl_indices_to_attack:
                if cls[:, i, :] == torch.tensor(params["class_to_replace"]):
                    found_required_class = True
                    break
            if not found_required_class:
                continue

            # Build new labels with target class substituted
            new_cls = cls.clone()
            patch_coords = []
            for bbox_idx in tl_indices_to_attack:
                if cls[:, bbox_idx, :] != torch.tensor(params["class_to_replace"]):
                    continue
                new_cls[:, bbox_idx, :] = torch.tensor(params["target_class"])

                current_bbox = bbox_coords[bbox_idx]
                x_center   = current_bbox[0].item() * img_size
                y_center   = current_bbox[1].item() * img_size
                bbox_width  = current_bbox[2].item() * img_size
                bbox_height = current_bbox[3].item() * img_size
                patch_start_y = int(y_center + bbox_height / 2)
                patch_coords.append((x_center, bbox_width, patch_start_y))

            num_bboxes = cls.shape[1]

            # ---------------------------------------------------------- #
            # PGD steps — delta must stay in the computation graph        #
            # ---------------------------------------------------------- #
            for pgd_step in range(params["pgd_steps"]):
                adv_img = img.clone()
                patch_placed = False

                for (x_center, bbox_width, patch_start_y) in patch_coords:
                    current_patch_size = int(bbox_width * int(params["patch_width_multiplier"]))
                    current_patch_start_x = int(x_center - current_patch_size / 2)

                    # -------------------------------------------------- #
                    # FIX: resize delta directly — gradient flows to delta #
                    # Do NOT use transform_patch here for loss computation #
                    # -------------------------------------------------- #
                    current_patch = patch_resize(delta, (current_patch_size, current_patch_size))

                    patch_end_x = current_patch_start_x + current_patch_size
                    patch_end_y = patch_start_y + current_patch_size

                    if (patch_end_x > img_size or patch_end_y > img_size
                            or current_patch_start_x < 0 or patch_start_y < 0):
                        continue

                    for i in range(img.shape[0]):
                        adv_img[i, :, patch_start_y:patch_end_y, current_patch_start_x:patch_end_x] = current_patch

                    patch_placed = True

                if not patch_placed:
                    continue

                # Forward pass on adversarial image
                result_perturbed = model.model(adv_img)
                train_batch = {
                    'cls': new_cls.to(device),
                    'bboxes': bbox_coords.to(device),
                    'batch_idx': torch.tensor([epoch]).repeat(num_bboxes).to(device)
                }

                attack_loss = v8loss(result_perturbed, train_batch)

                # -------------------------------------------------- #
                # FIX: TV loss on delta, not current_patch            #
                # -------------------------------------------------- #
                delta_batch = torch.unsqueeze(delta, dim=0)
                tv_loss = total_variation_loss(delta_batch)

                loss = attack_loss + params["tv_coef"] * tv_loss
                epoch_loss += loss.item()

                # Backprop — gradient reaches delta
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Project delta back to valid image range
                delta.data = torch.clamp(delta.data, min=0, max=1)

                # Save patch snapshot every 4 steps for evolution plot
                if step % 4 == 0:
                    save_image(delta.detach(), f'./{saved_patch_i}.png')
                    saved_patch_i += 1

                wandb.log({
                    "total_loss": loss.item(),
                    "attack_loss": attack_loss.item(),
                    "tv_loss": tv_loss.item(),
                    "step": step,
                })

                step += 1

        mean_epoch_loss = epoch_loss / max(1, params["pgd_steps"] * nb)
        print(f"Epoch {epoch+1} mean loss: {mean_epoch_loss:.4f}")
        wandb.log({"mean_epoch_loss": mean_epoch_loss, "epoch": epoch + 1})

        # Evaluate and save patch at end of each epoch
        if epoch % params["save_freq"] == 0:
            model.model.eval()
            print(f"\nRunning attack effectiveness evaluation (epoch {epoch+1})...")
            attack_effectiveness(params, delta.detach(), device, model, val_loader, "val", epoch)

            patch_img = wandb.Image(delta.detach().cpu().permute(1, 2, 0).numpy())
            wandb.log({"patch": patch_img})

            torch.save(delta.detach().cpu(), params["patch_name"] + str(epoch + 1) + ".pt")
            print(f"Patch saved: {params['patch_name']}{epoch+1}.pt")

            model.model.train()


def attack_effectiveness(params, patch, device, model, loader, name, epoch):
    """
    Evaluate how effective the current patch is on the validation set.
    Reports: attack success rate, vanishing rate, unaffected rate.
    """
    total_imgs_tested = 0
    imgs_attacked = 0
    imgs_unattacked = 0
    imgs_vanishing = 0

    img_to_store = None
    img_attacked_to_store = None

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        val_img = batch["img"].to(device).float() / 255.0
        cls = batch["cls"]
        bbox_coords = batch["bboxes"][0]

        if not torch.numel(bbox_coords):
            continue

        source_class = int(params["class_to_replace"])

        source_indices = [
            i
            for i in range(len(bbox_coords))
            if int(cls[0, i, 0].item()) == source_class
        ]

        if not source_indices:
            continue

        img_to_store = val_img.clone()
        img_attacked_to_store = val_img.clone()

        patch_placed = False

        for bbox_idx in source_indices:
            current_bbox = bbox_coords[bbox_idx]

            x_center = (
                current_bbox[0].item()
                * params["img_size"]
            )
            y_center = (
                current_bbox[1].item()
                * params["img_size"]
            )
            bbox_width = (
                current_bbox[2].item()
                * params["img_size"]
            )
            bbox_height = (
                current_bbox[3].item()
                * params["img_size"]
            )

            patch_size = int(
                bbox_width
                * float(
                    params["patch_width_multiplier"]
                )
            )

            if patch_size <= 0:
                continue

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
                patch_end_x > params["img_size"]
                or patch_end_y > params["img_size"]
                or patch_start_x < 0
                or patch_start_y < 0
            ):
                continue

            current_patch = patch_resize(
                patch,
                (patch_size, patch_size),
            )

            for i in range(
                img_attacked_to_store.shape[0]
            ):
                img_attacked_to_store[
                    i,
                    :,
                    patch_start_y:patch_end_y,
                    patch_start_x:patch_end_x,
                ] = current_patch

            patch_placed = True

        if not patch_placed:
            continue

        with torch.no_grad():
            result_val = model.predict(img_attacked_to_store, save=False, conf=0.25, verbose=False)
            bboxes = result_val[0].boxes
            if bboxes.cls.shape[0] == 0:
                imgs_vanishing += 1
            else:
                y_coords_pred = bboxes.xywh[:, 1]
                top_idx = torch.argmax(y_coords_pred)
                if bboxes.cls[top_idx] == torch.tensor(params["target_class"]):
                    imgs_attacked += 1
                else:
                    imgs_unattacked += 1

        total_imgs_tested += 1

    if total_imgs_tested == 0:
        print("WARNING: No images with the target class found in validation set!")
        return

    success_rate  = imgs_attacked   / total_imgs_tested * 100
    vanish_rate   = imgs_vanishing  / total_imgs_tested * 100
    unaffected    = imgs_unattacked / total_imgs_tested * 100

    print(f"\n--- Epoch {epoch+1} Attack Effectiveness ---")
    print(f"Total tested:       {total_imgs_tested}")
    print(f"Successful attacks: {imgs_attacked}  ({success_rate:.1f}%)")
    print(f"Vanishing:          {imgs_vanishing} ({vanish_rate:.1f}%)")
    print(f"Unaffected:         {imgs_unattacked} ({unaffected:.1f}%)")
    print(f"Total affected:     {imgs_attacked + imgs_vanishing} ({success_rate + vanish_rate:.1f}%)")

    wandb.log({
        f"attack_success_{name}":  success_rate,
        f"vanishing_{name}":       vanish_rate,
        f"unaffected_{name}":      unaffected,
        "epoch": epoch + 1,
    })

    # Log sample images to wandb
    if img_to_store is not None:
        result_clean = model.predict(img_to_store, save=False, conf=0.25, verbose=False)
        clean_annotated = result_clean[0].plot()[:, :, ::-1]
        wandb.log({f"Clean_{name}": wandb.Image(clean_annotated)})

    if img_attacked_to_store is not None:
        result_adv = model.predict(img_attacked_to_store, save=False, conf=0.25, verbose=False)
        adv_annotated = result_adv[0].plot()[:, :, ::-1]
        wandb.log({f"Attacked_{name}": wandb.Image(adv_annotated)})


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('\nUsage: python train_patch_yolov8_fixed.py <config.yaml>\n'
              'Example: python train_patch_yolov8_fixed.py configs/lisa_remapped_attack_fast.yaml')
        sys.exit(1)

    config_file_path = sys.argv[1]
    print(f'Starting fixed PGD attack with config: {config_file_path}')

    with open(config_file_path, 'r') as stream:
        try:
            wandb.init(project="tlr-attacks", reinit=True)
            config = yaml.safe_load(stream)
            wandb.config.update(config)
            wandb.config.update({"machine_name": platform.node()})
            params = wandb.config

            wandb.run.name = (
                datetime.now().strftime("%Y%m%d_%H%M%S")
                + "_FIXED_patch_"
                + params["dataset"]
                + "_"
                + str(params["img_size"])
            )

            random_seed = params.get("random_seed", 1)
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

            run_attack(params)

        except yaml.YAMLError as exc:
            print(exc)
