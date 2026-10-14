from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError("Configuration root must be a mapping.")
    validate_config(config)
    config["_config_path"] = str(config_path)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for section in ("project", "runtime", "protocol", "matrix"):
        if section not in config:
            raise ConfigError(f"Missing required section: {section}")

    matrix = config["matrix"]
    if not isinstance(matrix, list) or not matrix:
        raise ConfigError("matrix must be a non-empty list.")

    allowed_datasets = {"cifar10", "fashion_mnist"}
    allowed_models = {"smallcnn", "resnet18_small"}
    seen: set[tuple[str, str]] = set()
    for item in matrix:
        if not isinstance(item, dict):
            raise ConfigError("Each matrix entry must be a mapping.")
        dataset = item.get("dataset")
        model = item.get("model")
        if dataset not in allowed_datasets:
            raise ConfigError(f"Unsupported dataset: {dataset}")
        if model not in allowed_models:
            raise ConfigError(f"Unsupported model: {model}")
        key = (dataset, model)
        if key in seen:
            raise ConfigError(f"Duplicate matrix entry: {key}")
        seen.add(key)

    runtime = config["runtime"]
    for name in ("batch_size", "epochs", "main_runs", "seed_runs_per_seed", "perturbation_runs"):
        if int(runtime[name]) <= 0:
            raise ConfigError(f"{name} must be positive.")


def config_fingerprint(config: dict[str, Any]) -> str:
    payload = {key: value for key, value in config.items() if not key.startswith("_")}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
