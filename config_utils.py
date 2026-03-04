import json
import os
from typing import Any, Dict


def load_merged_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Loads the main config and merges in dataset-specific config.

    Rules:
    - Reads `dataset_format` from the main config (default: "voc").
    - Loads dataset config from `dataset_config_path` if provided, otherwise
      uses `{dataset_format}.json` in the repo root (e.g. `coco.json`, `voc.json`).
    - Shallow-merge: dataset config keys override main config keys.
    - If the dataset config file is missing, returns the main config unchanged.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        config: Dict[str, Any] = json.load(file)

    dataset_format = str(config.get("dataset_format", "voc")).strip().lower()
    dataset_config_path = str(
        config.get("dataset_config_path", f"{dataset_format}.json")
    ).strip()

    config_dir = os.path.dirname(os.path.abspath(config_path)) or os.getcwd()
    if dataset_config_path and not os.path.isabs(dataset_config_path):
        dataset_config_path = os.path.join(config_dir, dataset_config_path)

    if dataset_config_path and os.path.isfile(dataset_config_path):
        with open(dataset_config_path, "r", encoding="utf-8") as file:
            dataset_cfg = json.load(file)
        if isinstance(dataset_cfg, dict):
            merged = dict(config)
            merged.update(dataset_cfg)
            return merged

    return config
