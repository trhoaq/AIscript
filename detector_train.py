import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm

from data_loader import PascalVOCDataset, get_train_transforms, get_eval_transforms
from ssd_custom import SSD512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("config.json", "r", encoding="utf-8") as f:
    CFG = json.load(f)

DATA_ROOT = CFG["data_root"]
IMG_SIZE = CFG["img_size"]
BATCH_SIZE = CFG["batch_size"]
NUM_CLASSES = CFG["num_classes"]
LR = CFG["lr"]
WEIGHT_DECAY = CFG["weight_decay"]
EPOCHS = CFG["epochs"]
LOG_FILE = CFG["log_file"]
SCORE_THRESH = CFG["score_thresh"]
IOU_THRESH = CFG["iou_thresh"]
NMS_THRESH = CFG["nms_thresh"]


def collate_fn(batch):
    images = []
    targets = []
    for img, tgt in batch:
        boxes = torch.as_tensor(tgt["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(tgt["labels"], dtype=torch.int64) + 1
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([len(targets)]),
            "area": area,
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
        }
        images.append(img)
        targets.append(target)
    return images, targets


def voc07_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        if np.sum(rec >= t) == 0:
            p = 0.0
        else:
            p = np.max(prec[rec >= t])
        ap += p / 11.0
    return float(ap)


def evaluate_detection(model: nn.Module, data_loader: DataLoader, device: torch.device, num_classes: int, iou_thresh: float = 0.5, score_thresh: float = 0.05, ) -> Dict[str, float]:
    model.eval()
    class_gt_counts = [0 for _ in range(num_classes)]
    class_scores: List[List[float]] = [[] for _ in range(num_classes)]
    class_tp: List[List[int]] = [[] for _ in range(num_classes)]
    total_tp = 0
    total_fp = 0
    total_fn = 0
    iou_sum = 0.0
    iou_count = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for out, tgt in zip(outputs, targets):
                gt_boxes = tgt["boxes"].to(device)
                gt_labels = tgt["labels"].to(device)

                for c in range(1, num_classes):
                    gt_mask = gt_labels == c
                    gt_c = gt_boxes[gt_mask]
                    class_gt_counts[c] += gt_c.shape[0]

                pred_boxes = out["boxes"]
                pred_labels = out["labels"]
                pred_scores = out["scores"]

                keep = pred_scores >= score_thresh
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]
                pred_scores = pred_scores[keep]

                for c in range(1, num_classes):
                    gt_mask = gt_labels == c
                    gt_c = gt_boxes[gt_mask]
                    detected = torch.zeros((gt_c.shape[0],), dtype=torch.bool, device=device)

                    pred_mask = pred_labels == c
                    boxes_c = pred_boxes[pred_mask]
                    scores_c = pred_scores[pred_mask]
                    if boxes_c.numel() == 0:
                        continue

                    order = torch.argsort(scores_c, descending=True)
                    boxes_c = boxes_c[order]
                    scores_c = scores_c[order]

                    if gt_c.numel() == 0:
                        for s in scores_c:
                            class_scores[c].append(float(s))
                            class_tp[c].append(0)
                            total_fp += 1
                        continue

                    ious = box_iou(boxes_c, gt_c)
                    for i in range(boxes_c.shape[0]):
                        max_iou, max_idx = torch.max(ious[i], dim=0)
                        class_scores[c].append(float(scores_c[i]))
                        if max_iou >= iou_thresh and not detected[max_idx]:
                            detected[max_idx] = True
                            class_tp[c].append(1)
                            total_tp += 1
                            iou_sum += float(max_iou)
                            iou_count += 1
                        else:
                            class_tp[c].append(0)
                            total_fp += 1

                    total_fn += int(gt_c.shape[0] - detected.sum().item())

    ap_list = []
    for c in range(1, num_classes):
        scores = np.array(class_scores[c], dtype=np.float32)
        tps = np.array(class_tp[c], dtype=np.int32)
        n_gt = class_gt_counts[c]
        if n_gt == 0:
            continue
        if scores.size == 0:
            ap_list.append(0.0)
            continue
        order = scores.argsort()[::-1]
        tps = tps[order]
        fps = 1 - tps
        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        rec = tp_cum / max(1, n_gt)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1)
        ap_list.append(voc07_ap(rec, prec))

    mAP = float(np.mean(ap_list)) if ap_list else 0.0
    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    mean_iou = iou_sum / max(1, iou_count)

    return {
        "mAP@0.5": mAP,
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
    }


def main() -> None:
    train_ds = PascalVOCDataset(DATA_ROOT, "default", transform=get_train_transforms(IMG_SIZE))
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn
    )

    val_set_file = os.path.join(DATA_ROOT, "ImageSets", "Main", "val.txt")
    val_loader = None
    if os.path.exists(val_set_file):
        val_ds = PascalVOCDataset(DATA_ROOT, "val", transform=get_eval_transforms(IMG_SIZE))
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn
        )

    aspect_ratios = CFG["aspect_ratios"]
    model = SSD512(
        num_classes=NUM_CLASSES,
        aspect_ratios=aspect_ratios,
        img_size=IMG_SIZE,
        s_min=CFG["s_min"],
        s_max=CFG["s_max"],
        score_thresh=SCORE_THRESH,
        nms_thresh=NMS_THRESH,
    )
    model = model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    best_map = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]", dynamic_ncols=True)
        for images, targets in pbar:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loc_loss = loss_dict.get("bbox_regression")
            cls_loss = loss_dict.get("classification")
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            losses.backward()
            optimizer.step()

            total_loss += float(losses.item())
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{total_loss / max(1, len(pbar)):.4f}",
                loc=f"{float(loc_loss):.4f}" if loc_loss is not None else "n/a",
                cls=f"{float(cls_loss):.4f}" if cls_loss is not None else "n/a",
                lr=f"{lr:.3e}",
            )

        lr_scheduler.step()

        if val_loader is not None:
            metrics = evaluate_detection(
                model, val_loader, DEVICE, NUM_CLASSES, iou_thresh=IOU_THRESH, score_thresh=SCORE_THRESH
            )
            line = (
                f"epoch={epoch+1}/{EPOCHS}, "
                f"train_loss={total_loss/len(train_loader):.4f}, "
                f"loc_loss={float(loc_loss):.4f}, "
                f"cls_loss={float(cls_loss):.4f}, "
                f"mAP@0.5={metrics['mAP@0.5']:.4f}, "
                f"precision={metrics['precision']:.4f}, "
                f"recall={metrics['recall']:.4f}, "
                f"mean_iou={metrics['mean_iou']:.4f}, "
                f"lr={lr:.6f}\n"
            )
            print(line.strip())
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line)
            if metrics["mAP@0.5"] > best_map:
                best_map = metrics["mAP@0.5"]
                torch.save(model.state_dict(), CFG["ckpt_best"])
        else:
            line = (
                f"epoch={epoch+1}/{EPOCHS}, "
                f"train_loss={total_loss/len(train_loader):.4f}, "
                f"lr={lr:.6f}\n"
            )
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line)

    torch.save(model.state_dict(), CFG["ckpt_final"])
    print("Training Complete!")


if __name__ == "__main__":
    main()
