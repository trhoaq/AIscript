import argparse
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
_FALLBACK_BY_ORDER_WARNED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime webcam inference with OpenVINO INT8 model.")
    parser.add_argument("--config", default="config/config.json", help="Path to main config file.")
    parser.add_argument("--model", choices=["student", "teacher"], default="student", help="Model variant.")
    parser.add_argument("--ov-model", default=None, help="Path to OpenVINO IR .xml model. Default: models/openvino/{model}_int8.xml")
    parser.add_argument("--cam-id", type=str, default=0, help="Camera device id.")
    parser.add_argument("--device", default="CPU", help="OpenVINO device for runtime.")
    parser.add_argument("--score-thresh", type=float, default=None, help="Override score threshold.")
    parser.add_argument("--pre-nms-topk", type=int, default=None, help="Override pre-NMS top-k.")
    parser.add_argument("--max-detections", type=int, default=None, help="Override max detections per frame.")
    parser.add_argument("--line-thickness", type=int, default=2, help="Bounding box thickness.")
    parser.add_argument("--log-interval-sec", type=float, default=1.0, help="Terminal benchmark log interval in seconds.")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV window and run terminal-only benchmark.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 means run until interrupted).")
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


def _print_benchmark_log(
    total_frames: int,
    total_elapsed: float,
    window_frames: int,
    window_elapsed: float,
    window_pre_s: float,
    window_inf_s: float,
    window_post_s: float,
    window_total_s: float,
    window_det_count: int,
) -> None:
    if window_frames <= 0:
        return

    window_fps = window_frames / max(window_elapsed, 1e-6)
    avg_fps = total_frames / max(total_elapsed, 1e-6)
    avg_pre_ms = (window_pre_s / window_frames) * 1000.0
    avg_inf_ms = (window_inf_s / window_frames) * 1000.0
    avg_post_ms = (window_post_s / window_frames) * 1000.0
    avg_total_ms = (window_total_s / window_frames) * 1000.0
    avg_det_per_frame = window_det_count / max(window_frames, 1)

    print(
        "[Benchmark] "
        f"frames={total_frames} "
        f"window_fps={window_fps:.2f} "
        f"avg_fps={avg_fps:.2f} "
        f"latency_ms(total/pre/inf/post)="
        f"{avg_total_ms:.2f}/{avg_pre_ms:.2f}/{avg_inf_ms:.2f}/{avg_post_ms:.2f} "
        f"det_per_frame={avg_det_per_frame:.2f}"
    )


def _extract_logits_and_boxes(raw_outputs: Any, num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    named_arrays: List[Tuple[str, np.ndarray]] = []

    def _key_to_name(key: Any) -> str:
        if key is None:
            return ""
        if isinstance(key, str):
            return key
        try:
            if hasattr(key, "get_any_name"):
                name = key.get_any_name()
                if name:
                    return str(name)
        except Exception:
            pass
        try:
            if hasattr(key, "get_names"):
                names = list(key.get_names())
                if names:
                    return str(names[0])
        except Exception:
            pass
        return str(key)

    def _as_numpy(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        try:
            return np.asarray(value)
        except Exception:
            return np.array(value, dtype=object)

    def _collect(value: Any, name_hint: str = "") -> None:
        if value is None:
            return

        if isinstance(value, dict):
            for k, v in value.items():
                _collect(v, _key_to_name(k))
            return

        if hasattr(value, "items") and callable(getattr(value, "items")) and not isinstance(value, np.ndarray):
            try:
                items = list(value.items())
                if items:
                    for k, v in items:
                        _collect(v, _key_to_name(k))
                    return
            except Exception:
                pass

        if isinstance(value, (list, tuple)):
            for idx, v in enumerate(value):
                _collect(v, f"{name_hint}[{idx}]")
            return

        if hasattr(value, "values") and callable(getattr(value, "values")) and not isinstance(value, np.ndarray):
            try:
                vals = list(value.values())
                if vals:
                    for idx, v in enumerate(vals):
                        _collect(v, f"{name_hint}[{idx}]")
                    return
            except Exception:
                pass

        arr = _as_numpy(value)
        # Some OpenVINO wrappers may produce object arrays (e.g. sequence outputs).
        if arr.dtype == object:
            unpacked = False
            for item in arr.ravel().tolist():
                if isinstance(item, (np.ndarray, list, tuple, dict)) or hasattr(item, "shape"):
                    _collect(item, name_hint)
                    unpacked = True
            if unpacked:
                return
        named_arrays.append((name_hint, arr))

    _collect(raw_outputs, "output")

    cls_logits = None
    box_reg = None

    normalized_outputs: List[Tuple[str, np.ndarray]] = []
    for name, arr in named_arrays:
        if arr.ndim == 2:
            arr = arr[None, ...]

        if arr.ndim == 3 and arr.shape[1] == num_classes:
            arr = np.transpose(arr, (0, 2, 1))
        elif arr.ndim == 3 and arr.shape[1] == 4 and arr.shape[-1] != 4:
            arr = np.transpose(arr, (0, 2, 1))

        if arr.ndim == 3:
            normalized_outputs.append((name, arr.astype(np.float32)))

    # 1) Prefer output names when available.
    for name, arr in normalized_outputs:
        lname = name.lower()
        if cls_logits is None and arr.shape[-1] == num_classes and any(t in lname for t in ("cls", "class", "logit", "conf", "score")):
            cls_logits = arr
        if box_reg is None and arr.shape[-1] == 4 and any(t in lname for t in ("bbox", "box", "reg", "loc")):
            box_reg = arr

    if (cls_logits is None or box_reg is None) and len(normalized_outputs) >= 2:
        if normalized_outputs[0][1].shape[-1] == 4 and normalized_outputs[1][1].shape[-1] == 4:
            global _FALLBACK_BY_ORDER_WARNED
            if not _FALLBACK_BY_ORDER_WARNED:
                print("\n[CẢNH BÁO] Không thể phân biệt cls_logits và bbox_regression do shape giống nhau.")
                print("[CẢNH BÁO] Đang gán cứng thứ tự index: [0]=cls, [1]=bbox.\n")
                _FALLBACK_BY_ORDER_WARNED = True
            return normalized_outputs[0][1], normalized_outputs[1][1]

    # 2) Use shape heuristics when unambiguous.
    if cls_logits is None:
        cls_candidates = [arr for _, arr in normalized_outputs if arr.shape[-1] == num_classes]
        if num_classes != 4 and cls_candidates:
            cls_logits = cls_candidates[0]

    if box_reg is None:
        reg_candidates = [arr for _, arr in normalized_outputs if arr.shape[-1] == 4]
        if reg_candidates:
            if cls_logits is None:
                box_reg = reg_candidates[0]
            else:
                for cand in reg_candidates:
                    if cand is not cls_logits:
                        box_reg = cand
                        break
                if box_reg is None:
                    box_reg = reg_candidates[0]

    if cls_logits is None or box_reg is None:
        shapes = [(name, tuple(arr.shape)) for name, arr in normalized_outputs]
        raise RuntimeError(
            "Cannot identify OpenVINO outputs for cls_logits and bbox_regression. "
            f"Output shapes: {shapes}"
        )
    return cls_logits, box_reg
    
def main() -> None:
    args = parse_args()
    config = load_merged_config(args.config)

    if "obj_classes" not in config or not isinstance(config["obj_classes"], list):
        raise ValueError("obj_classes is missing or invalid in merged config.")

    num_classes = len(config["obj_classes"])
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

    if args.cam_id == "0":
        args.cam_id = int(args.cam_id)
    else:
        args.cam_id = str(args.cam_id)
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera with id={args.cam_id}")

    print(f"Using OpenVINO model: {ov_model_path}")
    print("Runtime config: PERFORMANCE_HINT=LATENCY, INFERENCE_NUM_THREADS=2")
    print(
        f"Benchmark config: input_size={img_size}, score_thresh={score_thresh}, "
        f"pre_nms_topk={pre_nms_topk}, max_detections={max_detections}, "
        f"log_interval_sec={max(0.0, float(args.log_interval_sec)):.2f}"
    )
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = float(cap.get(cv2.CAP_PROP_FPS))
    print(f"Camera info: id={args.cam_id}, resolution={cam_w}x{cam_h}, reported_fps={cam_fps:.2f}")
    print(f"Display: {'disabled' if args.no_display else 'enabled'}")
    print("Press 'q' or ESC to exit.")

    window_name = f"OpenVINO INT8 ({args.model}) - cam {args.cam_id}"
    display_enabled = not args.no_display
    if display_enabled:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            print(f"Warning: OpenCV GUI is unavailable. Falling back to no-display mode. detail={exc}")
            display_enabled = False
    run_start_time = time.perf_counter()
    last_log_time = run_start_time
    log_interval_sec = max(0.0, float(args.log_interval_sec))

    total_frames = 0
    total_pre_s = 0.0
    total_inf_s = 0.0
    total_post_s = 0.0
    total_total_s = 0.0
    total_det_count = 0

    window_frames = 0
    window_pre_s = 0.0
    window_inf_s = 0.0
    window_post_s = 0.0
    window_total_s = 0.0
    window_det_count = 0

    try:
        while True:
            frame_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Warning: Failed to read frame from camera. Exiting.")
                break

            pre_start = time.perf_counter()
            blob, meta = _preprocess_frame(frame, img_size)
            pre_end = time.perf_counter()

            inf_start = pre_end
            raw_outputs = compiled_model({input_name: blob})
            cls_logits_np, box_reg_np = _extract_logits_and_boxes(raw_outputs, num_classes)
            inf_end = time.perf_counter()

            post_start = inf_end
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
            detections_cpu = detections.cpu()
            _draw_detections(frame, detections_cpu, meta, args.line_thickness)
            post_end = time.perf_counter()

            frame_end = time.perf_counter()
            pre_s = pre_end - pre_start
            inf_s = inf_end - inf_start
            post_s = post_end - post_start
            frame_total_s = frame_end - frame_start

            det_count = int(detections_cpu.size(0))
            total_frames += 1
            total_pre_s += pre_s
            total_inf_s += inf_s
            total_post_s += post_s
            total_total_s += frame_total_s
            total_det_count += det_count

            window_frames += 1
            window_pre_s += pre_s
            window_inf_s += inf_s
            window_post_s += post_s
            window_total_s += frame_total_s
            window_det_count += det_count

            fps = 1.0 / max(frame_total_s, 1e-6)
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

            if display_enabled:
                try:
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break
                except cv2.error as exc:
                    print(f"Warning: OpenCV GUI call failed. Switching to no-display mode. detail={exc}")
                    display_enabled = False

            now = time.perf_counter()
            should_log = log_interval_sec == 0.0 or (now - last_log_time) >= log_interval_sec
            if should_log:
                _print_benchmark_log(
                    total_frames=total_frames,
                    total_elapsed=now - run_start_time,
                    window_frames=window_frames,
                    window_elapsed=now - last_log_time,
                    window_pre_s=window_pre_s,
                    window_inf_s=window_inf_s,
                    window_post_s=window_post_s,
                    window_total_s=window_total_s,
                    window_det_count=window_det_count,
                )
                last_log_time = now
                window_frames = 0
                window_pre_s = 0.0
                window_inf_s = 0.0
                window_post_s = 0.0
                window_total_s = 0.0
                window_det_count = 0

            if args.max_frames > 0 and total_frames >= int(args.max_frames):
                print(f"Reached max_frames={args.max_frames}. Stopping.")
                break
    finally:
        end_time = time.perf_counter()
        elapsed = end_time - run_start_time
        if total_frames > 0:
            avg_fps = total_frames / max(elapsed, 1e-6)
            avg_pre_ms = (total_pre_s / total_frames) * 1000.0
            avg_inf_ms = (total_inf_s / total_frames) * 1000.0
            avg_post_ms = (total_post_s / total_frames) * 1000.0
            avg_total_ms = (total_total_s / total_frames) * 1000.0
            avg_det = total_det_count / total_frames
            print(
                "[Summary] "
                f"frames={total_frames} "
                f"elapsed_sec={elapsed:.2f} "
                f"avg_fps={avg_fps:.2f} "
                f"avg_latency_ms(total/pre/inf/post)="
                f"{avg_total_ms:.2f}/{avg_pre_ms:.2f}/{avg_inf_ms:.2f}/{avg_post_ms:.2f} "
                f"avg_det_per_frame={avg_det:.2f}"
            )
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error as exc:
            print(f"Warning: cv2.destroyAllWindows() is unavailable in this environment. detail={exc}")


if __name__ == "__main__":
    main()
