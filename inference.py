import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch

from config_utils import load_merged_config
from model.ssdlite_ghostnet100 import SSDGhostNetV3
from model.ssdlite_mobilenet import SSDMobile


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime webcam inference with OpenVINO INT8 model.")
    parser.add_argument("--config", default="config/config.json", help="Path to main config file.")
    parser.add_argument("--model", choices=["student", "teacher"], default="student", help="Model variant.")
    parser.add_argument("--ov-model", default=None, help="Path to OpenVINO IR .xml model. Default: models/openvino/{model}_int8.xml")
    parser.add_argument("--cam-id", type=int, default=0, help="Camera device id.")
    parser.add_argument("--device", default="CPU", help="OpenVINO device for runtime.")
    parser.add_argument("--score-thresh", type=float, default=None, help="Override score threshold.")
    parser.add_argument("--pre-nms-topk", type=int, default=None, help="Override pre-NMS top-k.")
    parser.add_argument("--max-detections", type=int, default=None, help="Override max detections per frame.")
    parser.add_argument("--line-thickness", type=int, default=2, help="Bounding box thickness.")
    return parser.parse_args()


def _default_int8_path(model_kind: str) -> Path:
    return Path("models") / "openvino" / f"{model_kind}_int8.xml"


def _build_postprocess_model(config: Dict[str, Any], model_kind: str):
    num_classes = len(config["obj_classes"])
    img_size = int(config.get("img_size", 320))

    if model_kind == "student":
        width_mult = float(config.get("student_width", 1.0))
        model = SSDGhostNetV3(
            num_classes=num_classes,
            width_mult=width_mult,
            img_size=img_size,
            pretrained_backbone=False,
        )
    else:
        teacher_aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        model = SSDMobile(
            num_classes=num_classes,
            aspect_ratios=teacher_aspect_ratios,
            img_size=img_size,
            s_min=0.07,
            s_max=0.95,
            pretrained_backbone=False,
        )

    model.eval()
    return model


def _prepare_priors(post_model, img_size: int) -> torch.Tensor:
    dummy = torch.zeros((1, 3, img_size, img_size), dtype=torch.float32)
    with torch.no_grad():
        _, _, feats = post_model.forward_logits(dummy)
        priors = post_model.generate_priors(feats, dummy)
    return priors.cpu()


def _resolve_model_input_size(compiled_model, fallback_size: int) -> int:
    try:
        shape = list(compiled_model.input(0).shape)
    except Exception:
        return fallback_size

    if len(shape) == 4 and int(shape[2]) > 0 and int(shape[3]) > 0 and int(shape[2]) == int(shape[3]):
        return int(shape[2])
    return fallback_size


def _get_input_name(compiled_model) -> str:
    try:
        port = compiled_model.input(0)
    except Exception:
        port = compiled_model.inputs[0]

    try:
        name = port.get_any_name()
        if name:
            return name
    except Exception:
        pass

    try:
        names = list(port.get_names())
        if names:
            return names[0]
    except Exception:
        pass

    return "images"


def _preprocess_frame(frame_bgr: np.ndarray, img_size: int) -> Tuple[np.ndarray, Dict[str, float]]:
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
    norm = (rgb - MEAN) / STD
    nchw = np.transpose(norm, (2, 0, 1))[None, ...].astype(np.float32)

    meta = {
        "scale": float(scale),
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "orig_w": float(w),
        "orig_h": float(h),
    }
    return nchw, meta


def _map_box_to_original(box: torch.Tensor, meta: Dict[str, float]) -> Tuple[int, int, int, int]:
    scale = max(meta["scale"], 1e-6)
    pad_x = meta["pad_x"]
    pad_y = meta["pad_y"]
    w = meta["orig_w"]
    h = meta["orig_h"]

    x1, y1, x2, y2 = box.tolist()
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y2 = int(np.clip(y2, 0, h - 1))
    return x1, y1, x2, y2


def _draw_detections(
    frame: np.ndarray,
    detections: torch.Tensor,
    class_names: List[str],
    meta: Dict[str, float],
    line_thickness: int,
) -> None:
    if detections is None or detections.numel() == 0:
        return

    for det in detections:
        box = det[:4]

        x1, y1, x2, y2 = _map_box_to_original(box, meta)
        if x2 <= x1 or y2 <= y1:
            continue

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), radius=max(2, line_thickness * 2), color=(0, 255, 0), thickness=-1)


def _extract_logits_and_boxes(raw_outputs: Any, num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    arrays: List[np.ndarray] = []

    if isinstance(raw_outputs, dict):
        for value in raw_outputs.values():
            arrays.append(np.array(value))
    elif isinstance(raw_outputs, (list, tuple)):
        for value in raw_outputs:
            arrays.append(np.array(value))
    else:
        arrays.append(np.array(raw_outputs))

    cls_logits = None
    box_reg = None

    for arr in arrays:
        if arr.ndim == 3 and arr.shape[-1] == num_classes:
            cls_logits = arr
        elif arr.ndim == 3 and arr.shape[-1] == 4:
            box_reg = arr

    if cls_logits is None or box_reg is None:
        shapes = [tuple(a.shape) for a in arrays]
        raise RuntimeError(
            "Cannot identify OpenVINO outputs for cls_logits and bbox_regression. "
            f"Output shapes: {shapes}"
        )

    return cls_logits.astype(np.float32), box_reg.astype(np.float32)


def main() -> None:
    args = parse_args()
    config = load_merged_config(args.config)

    if "obj_classes" not in config or not isinstance(config["obj_classes"], list):
        raise ValueError("obj_classes is missing or invalid in merged config.")

    class_names = config["obj_classes"]
    num_classes = len(class_names)
    fallback_img_size = int(config.get("img_size", 320))

    ov_model_path = Path(args.ov_model) if args.ov_model else _default_int8_path(args.model)
    ov_model_path = ov_model_path.resolve()

    if not ov_model_path.exists():
        raise FileNotFoundError(
            f"Quantized OpenVINO model not found: {ov_model_path}\n"
            "Run export_onnx.py and quantize_openvino.py first."
        )

    try:
        import openvino as ov
    except ImportError as exc:
        raise RuntimeError(
            "OpenVINO is required for inference.py. Install in a supported Python environment (typically <= 3.12)."
        ) from exc

    core = ov.Core()
    ov_model = core.read_model(str(ov_model_path))

    runtime_config = {
        "PERFORMANCE_HINT": "LATENCY",
        "INFERENCE_NUM_THREADS": "2",
    }
    compiled_model = core.compile_model(ov_model, args.device, runtime_config)
    input_name = _get_input_name(compiled_model)

    img_size = _resolve_model_input_size(compiled_model, fallback_img_size)
    post_model = _build_postprocess_model(config, args.model)
    priors = _prepare_priors(post_model, img_size)

    score_thresh = float(args.score_thresh) if args.score_thresh is not None else float(config.get("score_thresh", 0.5))
    pre_nms_topk = int(args.pre_nms_topk) if args.pre_nms_topk is not None else int(config.get("eval_pre_nms_topk", 400))
    max_detections = int(args.max_detections) if args.max_detections is not None else int(config.get("eval_max_detections", 100))

    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera with id={args.cam_id}")

    print(f"Using OpenVINO model: {ov_model_path}")
    print("Runtime config: PERFORMANCE_HINT=LATENCY, INFERENCE_NUM_THREADS=2")
    print("Press 'q' or ESC to exit.")

    window_name = f"OpenVINO INT8 ({args.model}) - cam {args.cam_id}"
    prev_time = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Warning: Failed to read frame from camera. Exiting.")
                break

            blob, meta = _preprocess_frame(frame, img_size)
            raw_outputs = compiled_model({input_name: blob})
            cls_logits_np, box_reg_np = _extract_logits_and_boxes(raw_outputs, num_classes)

            cls_logits = torch.from_numpy(cls_logits_np)
            box_reg = torch.from_numpy(box_reg_np)

            with torch.no_grad():
                outputs = post_model.post_process(
                    cls_logits,
                    box_reg,
                    priors,
                    img_size=img_size,
                    score_thresh=score_thresh,
                    pre_nms_topk=pre_nms_topk,
                    max_detections=max_detections,
                )

            detections = outputs[0] if outputs else torch.zeros((0, 6), dtype=torch.float32)
            _draw_detections(frame, detections.cpu(), class_names, meta, args.line_thickness)

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (20, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
