import argparse
import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from config_utils import load_merged_config
from model.ssdlite_ghostnet100 import SSDGhostNetV3
from model.ssdlite_mobilenet import SSDMobile


class ONNXExportWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        cls_logits, box_reg, _ = self.model.forward_logits(images)
        return cls_logits, box_reg


def _torch_load_checkpoint(path: str, map_location):
    safe_globals = []
    try:
        import numpy as np

        np_core = getattr(np, "_core", None)
        if np_core is not None and hasattr(np_core, "multiarray") and hasattr(np_core.multiarray, "scalar"):
            safe_globals.append(np_core.multiarray.scalar)
        elif hasattr(np, "core") and hasattr(np.core, "multiarray") and hasattr(np.core.multiarray, "scalar"):
            safe_globals.append(np.core.multiarray.scalar)
    except Exception:
        pass

    try:
        if safe_globals and hasattr(torch.serialization, "safe_globals"):
            with torch.serialization.safe_globals(safe_globals):
                return torch.load(path, map_location=map_location, weights_only=True)
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception as exc:
        msg = str(exc)
        if "WeightsUnpickler error" in msg or "Unsupported global" in msg:
            print(
                "weights_only=True could not deserialize this checkpoint. "
                "Falling back to weights_only=False for trusted local file."
            )
            try:
                return torch.load(path, map_location=map_location, weights_only=False)
            except TypeError:
                return torch.load(path, map_location=map_location)
        raise


def _extract_model_state(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if state_dict is None and isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict")

    if state_dict is None and isinstance(checkpoint, dict) and checkpoint:
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError("No valid model_state_dict found in checkpoint.")

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    return state_dict


def _resolve_device(device_arg: str) -> torch.device:
    arg = str(device_arg).strip().lower()
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(arg)


def _build_model(config: Dict[str, Any], model_kind: str):
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
        ckpt_path = str(config.get("student_best_model_path", "")).strip()
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
        ckpt_path = str(config.get("teacher_best_model_path", "")).strip()

    if not ckpt_path:
        raise ValueError(f"Checkpoint path for model '{model_kind}' is empty in config.")

    return model, ckpt_path, img_size


def _default_output_path(model_kind: str) -> Path:
    output_dir = Path("models") / "openvino"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{model_kind}_fp32.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export student/teacher SSD model to ONNX (opset 18).")
    parser.add_argument("--config", default="config/config.json", help="Path to main config.")
    parser.add_argument("--model", choices=["student", "teacher"], default="student", help="Model variant.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint override.")
    parser.add_argument("--output", default=None, help="Output ONNX path.")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version.")
    parser.add_argument("--device", default="cpu", help="Export device: cpu|cuda|auto.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate model + checkpoint loading.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_merged_config(args.config)

    if "obj_classes" not in config or not isinstance(config["obj_classes"], list):
        raise ValueError("obj_classes is missing or invalid in merged config.")

    device = _resolve_device(args.device)
    model, ckpt_path, img_size = _build_model(config, args.model)

    if args.checkpoint:
        ckpt_path = str(args.checkpoint)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Please verify '{args.model}_best_model_path' in {args.config}."
        )

    checkpoint = _torch_load_checkpoint(ckpt_path, map_location=device)
    state_dict = _extract_model_state(checkpoint)

    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded checkpoint with strict=True: {ckpt_path}")
    except RuntimeError as exc:
        print(f"Strict loading failed: {exc}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            "Loaded checkpoint with strict=False. "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    model.to(device)
    model.eval()

    if args.dry_run:
        print("Dry-run completed. Model and checkpoint are ready for export.")
        return

    if importlib.util.find_spec("onnx") is None:
        raise RuntimeError(
            "Package 'onnx' is required to export ONNX. "
            "Install it in a supported Python environment (typically Python <= 3.12)."
        )

    output_path = Path(args.output) if args.output else _default_output_path(args.model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    export_model = ONNXExportWrapper(model).to(device)
    export_model.eval()

    torch.onnx.export(
        export_model,
        dummy_input,
        str(output_path),
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["images"],
        output_names=["cls_logits", "bbox_regression"],
        dynamic_axes={
            "images": {0: "batch"},
            "cls_logits": {0: "batch"},
            "bbox_regression": {0: "batch"},
        },
    )

    print(f"Exported ONNX model to: {output_path}")


if __name__ == "__main__":
    main()
