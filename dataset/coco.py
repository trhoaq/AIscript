import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2 # type: ignore
import numpy as np # type: ignore
from torch.utils.data import Dataset # type: ignore


def _load_default_obj_classes() -> List[str]:
    try:
        with open("config.json", "r", encoding="utf-8") as file:
            cfg = json.load(file)
        classes = cfg.get("obj_classes")
        if isinstance(classes, list) and classes:
            return classes
    except Exception:
        pass
    return ["background", "target"]


def _normalize_split_name(split: str) -> str:
    normalized = split[:-5] if split.endswith("_coco") else split
    aliases = {"train": "train2017", "val": "val2017"}
    return aliases.get(normalized, normalized)


def _resolve_images_dir(root: str, split: str) -> str:
    split = split.strip()
    normalized_split = _normalize_split_name(split)
    candidates = [
        os.path.join(root, split),
        os.path.join(root, "images"),
        os.path.join(root, "Images"),
        os.path.join(root, "JPEGImages"),
    ]
    if normalized_split != split:
        candidates.append(os.path.join(root, normalized_split))
    if split.endswith("_coco"):
        candidates.append(os.path.join(root, f"{normalized_split}_coco"))
    else:
        candidates.append(os.path.join(root, f"{split}_coco"))
        if normalized_split != split:
            candidates.append(os.path.join(root, f"{normalized_split}_coco"))

    # Common layout: root/images/<split>
    for images_dir_name in ("images", "Images", "JPEGImages"):
        candidates.append(os.path.join(root, images_dir_name, split))
        if normalized_split != split:
            candidates.append(os.path.join(root, images_dir_name, normalized_split))

    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if os.path.isdir(path):
            return path

    raise FileNotFoundError(
        f"Could not find image folder for split '{split}' under '{root}'. "
        f"Tried: {candidates}"
    )


def _resolve_annotation_path(root: str, split: str) -> str:
    split = split.strip()
    split_without_suffix = split[:-5] if split.endswith("_coco") else split
    normalized_split = _normalize_split_name(split)
    file_candidates = []
    for split_name in (split_without_suffix, normalized_split):
        candidate = f"instances_{split_name}.json"
        if candidate not in file_candidates:
            file_candidates.append(candidate)
    dir_candidates = [
        os.path.join(root, "annotations"),
        os.path.join(root, "annotations_trainval2017"),
        os.path.join(root, "annotations_trainval2017_coco"),
        root,
    ]

    tried_paths: List[str] = []
    for directory in dir_candidates:
        for filename in file_candidates:
            ann_path = os.path.join(directory, filename)
            tried_paths.append(ann_path)
            if os.path.isfile(ann_path):
                return ann_path

    raise FileNotFoundError(
        f"Could not find COCO annotations for split '{split}' under '{root}'. "
        f"Tried: {tried_paths}"
    )


class COCODataset(Dataset):
    """
    Dataset for COCO-style annotations (instances_*.json).
    Boxes are converted to pascal_voc format [xmin, ymin, xmax, ymax].
    """

    def __init__(
        self,
        root: str,
        split: str = "train2017",
        mode: str = "RGB",
        transform=None,
        obj_classes: Optional[List[str]] = None,
    ):
        self.root = root
        self.split = split
        self.mode = mode
        self.transform = transform

        self.obj_classes = obj_classes if obj_classes else _load_default_obj_classes()
        self.class_to_idx = {name: idx for idx, name in enumerate(self.obj_classes)}
        selected_class_names = set(self.obj_classes)
        if "background" in selected_class_names:
            selected_class_names.remove("background")

        self.images_dir = _resolve_images_dir(root, split)
        self.annotation_path = _resolve_annotation_path(root, split)

        with open(self.annotation_path, "r", encoding="utf-8") as file:
            coco_data = json.load(file)

        self.image_info: Dict[int, Dict[str, object]] = {
            int(img["id"]): img for img in coco_data.get("images", [])
        }

        self.category_id_to_label: Dict[int, int] = {}
        for category in coco_data.get("categories", []):
            cat_name = str(category.get("name", "")).strip()
            cat_id = int(category.get("id"))
            if cat_name in selected_class_names:
                self.category_id_to_label[cat_id] = self.class_to_idx[cat_name]

        if selected_class_names and not self.category_id_to_label:
            example_categories = ", ".join(
                str(category.get("name", "")).strip()
                for category in coco_data.get("categories", [])[:8]
            )
            raise ValueError(
                "No categories in obj_classes matched COCO categories. "
                f"split='{split}', annotation='{self.annotation_path}'. "
                f"Example categories in file: {example_categories}"
            )

        self.annotations_by_image: Dict[int, List[Tuple[List[float], int]]] = defaultdict(list)
        for ann in coco_data.get("annotations", []):
            if int(ann.get("iscrowd", 0)) == 1:
                continue

            label = self.category_id_to_label.get(int(ann.get("category_id", -1)))
            if label is None:
                continue

            image_id = int(ann.get("image_id", -1))
            if image_id not in self.image_info:
                continue

            bbox = ann.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            self.annotations_by_image[image_id].append((bbox, label))

        self.samples: List[Tuple[int, str]] = []
        for image_id, info in self.image_info.items():
            if image_id not in self.annotations_by_image:
                continue

            file_name = str(info.get("file_name", ""))
            image_path = os.path.join(self.images_dir, file_name)
            if os.path.isfile(image_path):
                self.samples.append((image_id, image_path))

        print(
            f"Found {len(self.samples)} valid COCO samples in split '{split}' "
            f"(Root: {root})."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_id, image_path = self.samples[index]

        image = cv2.imread(image_path)
        if image is None:
            return None, None

        if self.mode == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]

        boxes: List[List[float]] = []
        labels: List[int] = []
        for bbox, label in self.annotations_by_image.get(image_id, []):
            x, y, box_w, box_h = [float(v) for v in bbox]
            xmin = max(0.0, min(x, width - 1))
            ymin = max(0.0, min(y, height - 1))
            xmax = max(0.0, min(x + box_w, width - 1))
            ymax = max(0.0, min(y + box_h, height - 1))

            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        if not boxes:
            return None, None

        target = {
            "boxes": np.array(boxes, dtype=np.float32),
            "labels": np.array(labels, dtype=np.int64),
        }

        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=target["boxes"],
                class_labels=target["labels"],
                height=height,
                width=width,
            )
            image = transformed["image"]
            target["boxes"] = transformed["bboxes"]
            target["labels"] = transformed["class_labels"]

        return image, target


def get_coco_datasets(
    coco_root: str,
    img_size: int,
    transform_train=None,
    transform_val=None,
    train_split: str = "train2017",
    val_split: str = "val2017",
    obj_classes: Optional[List[str]] = None,
):
    """
    Helper function to get train and validation datasets for COCO format.
    img_size is kept for API compatibility with VOC helper.
    """
    _ = img_size

    try:
        train_ds = COCODataset(
            coco_root,
            split=train_split,
            transform=transform_train,
            obj_classes=obj_classes,
        )
    except FileNotFoundError as exc:
        if train_split != val_split:
            print(f"Warning: {exc}")
            print(f"Falling back to '{val_split}' for training split.")
            train_ds = COCODataset(
                coco_root,
                split=val_split,
                transform=transform_train,
                obj_classes=obj_classes,
            )
        else:
            raise

    try:
        val_ds = COCODataset(
            coco_root,
            split=val_split,
            transform=transform_val,
            obj_classes=obj_classes,
        )
    except FileNotFoundError as exc:
        if val_split != train_split:
            print(f"Warning: {exc}")
            print(f"Falling back to '{train_split}' for validation split.")
            val_ds = COCODataset(
                coco_root,
                split=train_split,
                transform=transform_val,
                obj_classes=obj_classes,
            )
        else:
            raise

    print(f"Total COCO train samples: {len(train_ds)}")
    print(f"Total COCO val samples: {len(val_ds)}")
    return train_ds, val_ds
