from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

PolicyAction = Literal["allow", "log", "block"]


@dataclass(frozen=True)
class PolicyProfile:
    name: str
    deterministic_algorithms: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    fill_uninitialized_memory: bool
    cublas_workspace_config: str | None
    on_violation: PolicyAction


def profile_for_mode(mode: str, cublas_workspace_config: str) -> PolicyProfile:
    if mode == "baseline":
        return PolicyProfile(
            name=mode,
            deterministic_algorithms=False,
            cudnn_benchmark=True,
            cudnn_deterministic=False,
            fill_uninitialized_memory=False,
            cublas_workspace_config=None,
            on_violation="allow",
        )
    if mode == "cei_lite":
        return PolicyProfile(
            name=mode,
            deterministic_algorithms=False,
            cudnn_benchmark=False,
            cudnn_deterministic=True,
            fill_uninitialized_memory=True,
            cublas_workspace_config=None,
            on_violation="log",
        )
    if mode == "cei_strict":
        return PolicyProfile(
            name=mode,
            deterministic_algorithms=True,
            cudnn_benchmark=False,
            cudnn_deterministic=True,
            fill_uninitialized_memory=True,
            cublas_workspace_config=cublas_workspace_config,
            on_violation="block",
        )
    if mode in {"perturbed", "perturbed_loader_seed", "perturbed_model_seed"}:
        return PolicyProfile(
            name=mode,
            deterministic_algorithms=True,
            cudnn_benchmark=False,
            cudnn_deterministic=True,
            fill_uninitialized_memory=True,
            cublas_workspace_config=cublas_workspace_config,
            on_violation="log",
        )
    raise ValueError(f"Unknown policy mode: {mode}")


def prepare_process_environment(profile: PolicyProfile) -> None:
    if profile.cublas_workspace_config:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = profile.cublas_workspace_config
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


class PolicyLogger:
    def __init__(self, path: Path, profile: PolicyProfile) -> None:
        self.path = path
        self.profile = profile
        self._event_index = 0

    def record(self, event: str, *, level: str = "info", **details: Any) -> None:
        payload = {
            "event_index": self._event_index,
            "event": event,
            "level": level,
            "policy_profile": self.profile.name,
            **details,
        }
        self._event_index += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def violation(self, event: str, **details: Any) -> None:
        action = self.profile.on_violation
        self.record(event, level="violation", action=action, **details)
        if action == "block":
            raise RuntimeError(f"Blocked policy violation: {event}")


def apply_policy(profile: PolicyProfile, logger: PolicyLogger) -> dict[str, Any]:
    torch.backends.cudnn.benchmark = bool(profile.cudnn_benchmark)
    torch.backends.cudnn.deterministic = bool(profile.cudnn_deterministic)
    torch.use_deterministic_algorithms(bool(profile.deterministic_algorithms))

    try:
        torch.utils.deterministic.fill_uninitialized_memory = bool(
            profile.fill_uninitialized_memory
        )
    except (AttributeError, RuntimeError):
        pass

    actual = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "profile": asdict(profile),
    }
    logger.record("apply_policy", **actual)
    return actual
