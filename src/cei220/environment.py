from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torchvision


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def nvidia_driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return lines[0] if lines else None
    except Exception:
        return None


def collect_environment() -> dict[str, Any]:
    stable = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "nvidia_driver": nvidia_driver_version(),
    }
    return {**stable, "environment_fingerprint": _stable_hash(stable)}


def check_expected_environment(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    if not expected:
        return {"passed": True, "checks": {}}

    checks = {
        "python_prefix": {
            "expected": expected.get("python_prefix"),
            "actual": actual.get("python"),
            "passed": str(actual.get("python", "")).startswith(str(expected.get("python_prefix", ""))),
        },
        "torch_prefix": {
            "expected": expected.get("torch_prefix"),
            "actual": actual.get("torch"),
            "passed": str(actual.get("torch", "")).startswith(str(expected.get("torch_prefix", ""))),
        },
        "torchvision_prefix": {
            "expected": expected.get("torchvision_prefix"),
            "actual": actual.get("torchvision"),
            "passed": str(actual.get("torchvision", "")).startswith(str(expected.get("torchvision_prefix", ""))),
        },
        "cuda": {
            "expected": str(expected.get("cuda")),
            "actual": str(actual.get("cuda_build")),
            "passed": str(actual.get("cuda_build")) == str(expected.get("cuda")),
        },
        "cudnn": {
            "expected": expected.get("cudnn"),
            "actual": actual.get("cudnn"),
            "passed": int(actual.get("cudnn") or -1) == int(expected.get("cudnn") or -2),
        },
        "gpu": {
            "expected_contains": expected.get("gpu_contains"),
            "actual": actual.get("gpu_name"),
            "passed": str(expected.get("gpu_contains", "")) in str(actual.get("gpu_name", "")),
        },
    }
    return {"passed": all(item["passed"] for item in checks.values()), "checks": checks}


def _iter_code_files(project_root: Path) -> Iterable[Path]:
    include_roots = [project_root / "src", project_root / "scripts", project_root / "configs"]
    fixed_files = [project_root / "pyproject.toml", project_root / "requirements-lock.txt"]
    for root in include_roots:
        if root.exists():
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".toml", ".txt"}:
                    yield path
    for path in fixed_files:
        if path.exists():
            yield path


def code_fingerprint(project_root: str | Path) -> str:
    root = Path(project_root).resolve()
    digest = hashlib.sha256()
    for path in _iter_code_files(root):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_commit(project_root: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(project_root),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or None
    except Exception:
        return None
