from typing import Dict, Tuple, Union

import cv2
import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_bgr_frame(
    frame_bgr: np.ndarray,
    img_size: int,
    return_meta: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
    """
    Letterbox-resize BGR frame to a square tensor and normalize to ImageNet stats.
    """
    h, w = frame_bgr.shape[:2]

    scale = min(img_size / max(w, 1), img_size / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    pad_x = (img_size - new_w) // 2
    pad_y = (img_size - new_h) // 2
    padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norm = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    nchw = np.transpose(norm, (2, 0, 1))[None, ...].astype(np.float32)

    if not return_meta:
        return nchw

    meta = {
        "scale": float(scale),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "orig_w": float(w),
        "orig_h": float(h),
    }
    return nchw, meta
