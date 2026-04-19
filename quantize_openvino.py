import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from core.config_utils import load_merged_config
from core.openvino_preprocess import preprocess_bgr_frame
from core.openvino_runtime_utils import get_openvino_input_name


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-training quantization with OpenVINO NNCF (KL-divergence style calibration).")
    parser.add_argument("--config", default="config/config.json", help="Path to main config.")
    parser.add_argument("--model", choices=["student", "teacher"], default="student", help="Model variant for naming outputs.")
    parser.add_argument("--onnx", default=None, help="Input ONNX model path. Default: models/openvino/{model}_fp32.onnx")
    parser.add_argument("--output-dir", default="models/openvino", help="Directory to store FP32/INT8 OpenVINO IR.")
    parser.add_argument("--calib-dir", default=None, help="Calibration image root. If omitted, auto-detect from dataset config.")
    parser.add_argument("--subset-size", type=int, default=300, help="Number of calibration samples.")
    parser.add_argument("--device", default="CPU", help="OpenVINO device for verification run.")
    parser.add_argument("--no-verify-runtime", action="store_true", help="Skip verification inference after quantization.")
    return parser.parse_args()


def _default_onnx_path(model_kind: str) -> Path:
    return Path("models") / "openvino" / f"{model_kind}_fp32.onnx"


def _collect_images(root: Path) -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _auto_calib_dirs(config: Dict[str, Any]) -> List[Path]:
    dataset_format = str(config.get("dataset_format", "voc")).strip().lower()
    dirs: List[Path] = []

    if dataset_format == "voc":
        voc_root = Path(str(config.get("voc_root", "./data"))).resolve()
        voc_years = config.get("voc_years", ["2012"])
        for year in voc_years:
            year_str = str(year)
            dirs.append(voc_root / "VOCdevkit" / f"VOC{year_str}" / "JPEGImages")
            dirs.append(voc_root / f"VOC{year_str}" / "JPEGImages")
        dirs.append(voc_root / "JPEGImages")
        dirs.append(voc_root)
    else:
        coco_root = Path(str(config.get("coco_root", "./data"))).resolve()
        train_split = str(config.get("coco_train_split", config.get("train_split", "train2017")))
        dirs.append(coco_root / train_split)
        dirs.append(coco_root)

    unique_dirs: List[Path] = []
    seen = set()
    for d in dirs:
        k = str(d)
        if k not in seen:
            unique_dirs.append(d)
            seen.add(k)
    return unique_dirs


def _gather_calibration_images(config: Dict[str, Any], calib_dir: Optional[str], subset_size: int) -> List[Path]:
    subset_size = max(1, int(subset_size))

    candidates: List[Path] = []
    if calib_dir:
        candidates.append(Path(calib_dir).resolve())
    else:
        candidates.extend(_auto_calib_dirs(config))

    all_images: List[Path] = []
    for directory in candidates:
        images = _collect_images(directory)
        if images:
            all_images.extend(images)
            if len(all_images) >= subset_size:
                break

    if not all_images:
        tried = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(
            "No calibration images found. Checked directories:\n"
            f"{tried}"
        )

    return all_images[:subset_size]


def _build_kl_advanced_parameters(nncf):
    try:
        from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
        from nncf.quantization.range_estimator import (
            AggregatorType,
            RangeEstimatorParameters,
            StatisticsCollectorParameters,
            StatisticsType,
        )

        histogram_collector = StatisticsCollectorParameters(
            statistics_type=StatisticsType.RAW,
            aggregator_type=AggregatorType.HISTOGRAM,
        )
        range_estimator = RangeEstimatorParameters(
            min=histogram_collector,
            max=histogram_collector,
        )
        return AdvancedQuantizationParameters(
            activations_range_estimator_params=range_estimator,
        )
    except Exception as exc:
        print(
            "Warning: Could not configure explicit KL/histogram range estimator "
            f"for this NNCF version ({exc}). Falling back to default PTQ calibration."
        )
        return None


def _quantize_with_fallback(nncf, ov_model, calibration_dataset, subset_size: int, adv_params):
    attempts = []

    base_kwargs: Dict[str, Any] = {"subset_size": subset_size}
    if hasattr(nncf, "QuantizationPreset"):
        base_kwargs["preset"] = nncf.QuantizationPreset.MIXED
    if adv_params is not None:
        base_kwargs["advanced_parameters"] = adv_params

    attempts.append(base_kwargs)

    kwargs_without_adv = dict(base_kwargs)
    kwargs_without_adv.pop("advanced_parameters", None)
    attempts.append(kwargs_without_adv)

    attempts.append({"subset_size": subset_size})
    attempts.append({})

    last_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            print(f"Trying nncf.quantize with kwargs: {kwargs}")
            return nncf.quantize(ov_model, calibration_dataset, **kwargs)
        except TypeError as exc:
            last_error = exc
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError("Unable to quantize model with current NNCF API.")


def _save_ov_model(ov, ov_model, xml_path: Path) -> None:
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(ov, "save_model"):
        ov.save_model(ov_model, str(xml_path))
    else:
        ov.serialize(ov_model, str(xml_path))


def main() -> None:
    args = parse_args()
    config = load_merged_config(args.config)
    img_size = int(config.get("img_size", 320))

    onnx_path = Path(args.onnx) if args.onnx else _default_onnx_path(args.model)
    onnx_path = onnx_path.resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {onnx_path}\n"
            "Run export_onnx.py first."
        )

    try:
        import openvino as ov
        import nncf
    except ImportError as exc:
        raise RuntimeError(
            "OpenVINO/NNCF is required for PTQ. Install in a supported Python environment (typically <= 3.12)."
        ) from exc

    image_paths = _gather_calibration_images(config, args.calib_dir, args.subset_size)
    calibration_inputs: List[np.ndarray] = []
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        calibration_inputs.append(preprocess_bgr_frame(image, img_size))

    if not calibration_inputs:
        raise RuntimeError("No valid calibration images could be loaded by OpenCV.")

    print(f"Loaded {len(calibration_inputs)} calibration samples.")

    core = ov.Core()
    ov_model = core.read_model(str(onnx_path))
    input_name = get_openvino_input_name(ov_model)

    calibration_dataset = nncf.Dataset(
        calibration_inputs,
        lambda blob: {input_name: blob},
    )

    adv_params = _build_kl_advanced_parameters(nncf)
    quantized_model = _quantize_with_fallback(
        nncf,
        ov_model,
        calibration_dataset,
        subset_size=min(len(calibration_inputs), max(1, int(args.subset_size))),
        adv_params=adv_params,
    )

    output_dir = Path(args.output_dir).resolve()
    fp32_xml = output_dir / f"{args.model}_fp32.xml"
    int8_xml = output_dir / f"{args.model}_int8.xml"

    _save_ov_model(ov, ov_model, fp32_xml)
    _save_ov_model(ov, quantized_model, int8_xml)

    print(f"Saved FP32 IR to: {fp32_xml}")
    print(f"Saved INT8 IR to: {int8_xml}")

    if not args.no_verify_runtime:
        runtime_config = {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_NUM_THREADS": "2",
        }
        compiled_model = core.compile_model(quantized_model, args.device, runtime_config)
        verify_input_name = get_openvino_input_name(compiled_model)
        _ = compiled_model({verify_input_name: calibration_inputs[0]})
        print(
            "Verified one OpenVINO inference with "
            "PERFORMANCE_HINT=LATENCY and INFERENCE_NUM_THREADS=2."
        )


if __name__ == "__main__":
    main()
