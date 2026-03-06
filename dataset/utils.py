import os
import random
from typing import List, Tuple

import numpy as np
import torch


def set_seed_everything(seed: int = 42, deterministic: bool = False) -> int:
    """
    Set random seed for Python, NumPy and PyTorch.

    Args:
        seed: Seed value to apply.
        deterministic: If True, force deterministic CUDA/cuDNN behavior where possible.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some operators do not support deterministic mode on every platform.
            pass

    print(f"Global seed set to {seed} (deterministic={bool(deterministic)})")
    return seed


def _resolve_voc_images_path(root: str) -> str:
    if os.path.exists(os.path.join(root, "JPEGImages")):
        return os.path.join(root, "JPEGImages")
    if os.path.exists(os.path.join(root, "Images")):
        return os.path.join(root, "Images")
    return os.path.join(root, "Image")


def _read_split_ids(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig") as file:
        return [line.strip() for line in file if line.strip()]


def _write_split_ids(path: str, ids: List[str]) -> None:
    unique_ids = list(dict.fromkeys(ids))
    with open(path, "w", encoding="utf-8") as file:
        if unique_ids:
            file.write("\n".join(unique_ids) + "\n")


def _collect_voc_ids(root: str) -> List[str]:
    images_path = _resolve_voc_images_path(root)
    anno_path = os.path.join(root, "Annotations")

    if not os.path.isdir(images_path) or not os.path.isdir(anno_path):
        return []

    image_ids = {
        os.path.splitext(name)[0]
        for name in os.listdir(images_path)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    }
    anno_ids = {
        os.path.splitext(name)[0]
        for name in os.listdir(anno_path)
        if name.lower().endswith(".xml")
    }

    return sorted(image_ids & anno_ids)


def _random_split_ids(all_ids: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    ids = list(all_ids)
    rng = random.Random(int(seed))
    rng.shuffle(ids)

    if len(ids) <= 1:
        return ids, ids

    ratio = float(val_ratio)
    ratio = max(0.01, min(0.99, ratio))
    val_count = int(round(len(ids) * ratio))
    val_count = max(1, min(val_count, len(ids) - 1))

    val_ids = ids[:val_count]
    train_ids = ids[val_count:]
    return train_ids, val_ids


def ensure_voc_train_val_split(
    root: str,
    train_split: str = "train",
    val_split: str = "val",
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Ensure VOC train/val split files exist.
    If split files are missing, automatically creates deterministic train/val IDs.
    """
    split_dir = os.path.join(root, "ImageSets", "Main")
    os.makedirs(split_dir, exist_ok=True)

    train_file = os.path.join(split_dir, f"{train_split}.txt")
    val_file = os.path.join(split_dir, f"{val_split}.txt")

    train_ids = _read_split_ids(train_file)
    val_ids = _read_split_ids(val_file)

    if train_ids and val_ids:
        return train_ids, val_ids

    all_ids = _collect_voc_ids(root)
    if not all_ids:
        raise FileNotFoundError(
            f"No valid VOC samples found under '{root}'. "
            "Expected images + annotations to auto-create train/val split."
        )

    if train_ids and not val_ids:
        train_set = set(train_ids)
        val_ids = [img_id for img_id in all_ids if img_id not in train_set]
        if not val_ids:
            _, val_ids = _random_split_ids(all_ids, val_ratio=val_ratio, seed=seed)
        val_ids = sorted(set(val_ids))
    elif val_ids and not train_ids:
        val_set = set(val_ids)
        train_ids = [img_id for img_id in all_ids if img_id not in val_set]
        if not train_ids:
            train_ids, _ = _random_split_ids(all_ids, val_ratio=val_ratio, seed=seed)
        train_ids = sorted(set(train_ids))
    else:
        train_ids, val_ids = _random_split_ids(all_ids, val_ratio=val_ratio, seed=seed)

    # Always keep canonical files for interoperability.
    canonical_train = os.path.join(split_dir, "train.txt")
    canonical_val = os.path.join(split_dir, "val.txt")
    canonical_trainval = os.path.join(split_dir, "trainval.txt")
    _write_split_ids(canonical_train, train_ids)
    _write_split_ids(canonical_val, val_ids)
    _write_split_ids(canonical_trainval, sorted(set(train_ids + val_ids)))

    # Keep requested split names in sync.
    _write_split_ids(train_file, train_ids)
    _write_split_ids(val_file, val_ids)

    print(
        f"Auto-created VOC split at '{root}': "
        f"train={len(train_ids)} val={len(val_ids)} seed={seed} val_ratio={val_ratio}"
    )
    return train_ids, val_ids
