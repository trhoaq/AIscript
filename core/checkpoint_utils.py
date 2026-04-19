import os
import re
from typing import Any, Dict, Optional

import torch


def torch_load_checkpoint(path: str, map_location) -> Any:
    """
    PyTorch 2.6-compatible checkpoint loading for trusted local artifacts.
    """
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


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def extract_model_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """
    Extract model state dict from common checkpoint layouts.
    """
    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint format: expected a dict.")

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        state_dict = checkpoint.get("state_dict")

    if state_dict is None and checkpoint:
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError("No valid model_state_dict found in checkpoint.")

    return _strip_module_prefix(state_dict)


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return None

    checkpoints = [
        os.path.join(checkpoint_dir, name)
        for name in os.listdir(checkpoint_dir)
        if name.lower().endswith(".pth")
    ]
    if not checkpoints:
        return None

    def sort_key(path: str):
        name = os.path.basename(path)
        match = re.match(r"epoch_(\d+)_(last|best)\.pth$", name)
        if match:
            epoch = int(match.group(1))
            kind_priority = 2 if match.group(2) == "last" else 1
            return (1, epoch, kind_priority, os.path.getmtime(path))
        return (0, 0, 0, os.path.getmtime(path))

    return max(checkpoints, key=sort_key)
