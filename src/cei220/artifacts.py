from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_directory(
    project_root: str | Path,
    config: dict[str, Any],
    *,
    dataset: str,
    model: str,
    group: str,
    mode: str,
    external_seed: int,
    run_index: int,
) -> Path:
    base = Path(project_root) / config["project"]["output_root"] / config["project"]["experiment_name"]
    return (
        base
        / dataset
        / model
        / group
        / mode
        / f"seed_{external_seed}"
        / f"run_{run_index:03d}"
    )


def result_root(project_root: str | Path, config: dict[str, Any]) -> Path:
    return Path(project_root) / config["project"]["output_root"] / config["project"]["experiment_name"]
