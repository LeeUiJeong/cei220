from __future__ import annotations

import json
import os
import random
import shutil
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim

from .artifacts import run_directory, write_json
from .config import config_fingerprint
from .data import build_loaders, dataset_spec
from .environment import code_fingerprint, collect_environment, git_commit
from .hashing import hash_state_dict, sha256_file
from .models import build_model
from .policy import PolicyLogger, apply_policy, profile_for_mode
from .seeds import SeedManager, set_all_rngs


def _train_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        loss_sum += float(loss.item()) * batch_size
        correct += int((outputs.argmax(dim=1) == targets).sum().item())
        total += batch_size
    return loss_sum / total, correct / total


@torch.no_grad()
def _evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        batch_size = targets.size(0)
        loss_sum += float(loss.item()) * batch_size
        correct += int((outputs.argmax(dim=1) == targets).sum().item())
        total += batch_size
    return loss_sum / total, correct / total


def _select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def run_job(
    *,
    project_root: str | Path,
    config: dict[str, Any],
    dataset: str,
    model_name: str,
    group: str,
    mode: str,
    external_seed: int,
    run_index: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    runtime = config["runtime"]
    project = config["project"]
    run_dir = run_directory(
        project_root,
        config,
        dataset=dataset,
        model=model_name,
        group=group,
        mode=mode,
        external_seed=external_seed,
        run_index=run_index,
    )

    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    status_path = run_dir / "run_status.json"
    write_json(
        status_path,
        {
            "status": "running",
            "dataset": dataset,
            "model": model_name,
            "group": group,
            "mode": mode,
            "external_seed": external_seed,
            "run_index": run_index,
        },
    )

    try:
        environment = collect_environment()
        code_hash = code_fingerprint(project_root)
        config_hash = config_fingerprint(config)
        data_manifest_path = project_root / project["data_root"] / "manifests" / f"{dataset}.json"
        data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))

        profile = profile_for_mode(mode, runtime["cublas_workspace_config"])
        policy_logger = PolicyLogger(run_dir / "policy_log.jsonl", profile)
        actual_policy = apply_policy(profile, policy_logger)

        seed_manager = SeedManager(
            external_seed=int(external_seed),
            dataset=dataset,
            model=model_name,
            namespace_version=project["namespace_version"],
            salt=project["hkdf_salt"],
            log_path=run_dir / "seed_transparency_log.jsonl",
        )

        if mode == "baseline":
            global_seed = model_seed = train_seed = worker_seed = eval_seed = augmentation_seed = int(external_seed)
            set_all_rngs(global_seed)
        else:
            global_seed = seed_manager.derive("global/python_numpy_torch", "global_rngs")
            set_all_rngs(global_seed)
            seed_manager.inject("global/python_numpy_torch", global_seed, "global_rngs")
            model_seed = seed_manager.derive("model/init", "model")
            train_seed = seed_manager.derive("data/loader/train", "train_loader")
            worker_seed = seed_manager.derive("data/loader/worker_base", "data_loader_workers")
            augmentation_seed = seed_manager.derive("augmentation/train", "augmentation")
            eval_seed = seed_manager.derive("evaluation/test", "evaluation")

        if mode == "perturbed":
            policy_logger.record(
                "intentional_log_only_perturbation",
                level="warning",
                description="An unmanaged NumPy generator was created but not used by training.",
            )
            _ = np.random.default_rng().random()

        if mode == "perturbed_model_seed":
            unmanaged_seed = (int(external_seed) + 1_000_003 * (run_index + 1)) % (2**32)
            policy_logger.violation(
                "unmanaged_model_seed_override",
                original_seed=int(model_seed),
                unmanaged_seed=int(unmanaged_seed),
            )
            model_seed = unmanaged_seed

        if mode == "perturbed_loader_seed":
            unmanaged_seed = (int(external_seed) + 2_000_003 * (run_index + 1)) % (2**32)
            policy_logger.violation(
                "unmanaged_loader_seed_override",
                original_seed=int(train_seed),
                unmanaged_seed=int(unmanaged_seed),
            )
            train_seed = unmanaged_seed

        device = _select_device(str(runtime["device"]))
        loaders = build_loaders(
            dataset,
            project_root / project["data_root"],
            batch_size=int(runtime["batch_size"]),
            num_workers=int(runtime["num_workers"]),
            pin_memory=bool(runtime["pin_memory"] and device.type == "cuda"),
            train_seed=int(train_seed),
            worker_seed=int(worker_seed),
            eval_seed=int(eval_seed),
        )
        train_loader, test_loader = loaders
        if mode != "baseline":
            seed_manager.inject("data/loader/train", int(train_seed), "train_loader_generator")
            if int(runtime["num_workers"]) > 0:
                seed_manager.inject(
                    "data/loader/worker_base",
                    int(worker_seed),
                    "data_loader_workers",
                )
            seed_manager.inject("evaluation/test", int(eval_seed), "evaluation_generator")

        spec = dataset_spec(dataset)
        set_all_rngs(int(model_seed))
        if mode != "baseline":
            seed_manager.inject("model/init", int(model_seed), "model")
        model = build_model(model_name, spec["in_channels"], spec["num_classes"]).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer_name = str(runtime["optimizer"]).lower()
        if optimizer_name != "adam":
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(runtime["learning_rate"]),
            weight_decay=float(runtime.get("weight_decay", 0.0)),
        )

        set_all_rngs(int(augmentation_seed))
        if mode != "baseline":
            seed_manager.inject("augmentation/train", int(augmentation_seed), "augmentation")

        started = time.perf_counter()
        history: list[dict[str, Any]] = []
        for epoch in range(int(runtime["epochs"])):
            train_loss, train_accuracy = _train_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, test_accuracy = _evaluate(model, test_loader, criterion, device)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                }
            )
        runtime_seconds = time.perf_counter() - started

        state_dict = model.state_dict()
        model_state_sha256 = hash_state_dict(state_dict)
        torch.save(state_dict, run_dir / "model_state_only.pt")
        torch.save(
            {
                "model_state_dict": state_dict,
                "dataset": dataset,
                "model": model_name,
                "group": group,
                "mode": mode,
                "external_seed": int(external_seed),
                "run_index": int(run_index),
                "config_fingerprint": config_hash,
            },
            run_dir / "checkpoint.pt",
        )

        metrics = {
            "dataset": dataset,
            "model": model_name,
            "group": group,
            "mode": mode,
            "external_seed": int(external_seed),
            "run_index": int(run_index),
            "runtime_seconds": runtime_seconds,
            "final_train_loss": history[-1]["train_loss"],
            "final_train_accuracy": history[-1]["train_accuracy"],
            "final_test_loss": history[-1]["test_loss"],
            "final_test_accuracy": history[-1]["test_accuracy"],
            "model_state_sha256": model_state_sha256,
            "history": history,
        }
        write_json(run_dir / "metrics.json", metrics)

        metadata = {
            "experiment_name": project["experiment_name"],
            "artifact_schema_version": project["artifact_schema_version"],
            "seed_derivation_version": project["seed_derivation_version"],
            "namespace_version": project["namespace_version"],
            "dataset": dataset,
            "model": model_name,
            "group": group,
            "mode": mode,
            "external_seed": int(external_seed),
            "run_index": int(run_index),
            "runtime_config": runtime,
            "policy_profile": asdict(profile),
            "actual_policy_state": actual_policy,
            "environment": environment,
            "code_fingerprint": code_hash,
            "config_fingerprint": config_hash,
            "git_commit": git_commit(project_root),
            "dataset_fingerprint": data_manifest["dataset_fingerprint"],
        }
        write_json(run_dir / "metadata.json", metadata)

        artifact_names = [
            "checkpoint.pt",
            "model_state_only.pt",
            "metrics.json",
            "metadata.json",
            "policy_log.jsonl",
            "seed_transparency_log.jsonl",
        ]
        hashes: dict[str, str] = {"__model_state_dict__": model_state_sha256}
        for name in artifact_names:
            path = run_dir / name
            if path.exists():
                hashes[name] = sha256_file(path)
        write_json(run_dir / "hashes.json", hashes)

        completed_status = {
            "status": "completed",
            "dataset": dataset,
            "model": model_name,
            "group": group,
            "mode": mode,
            "external_seed": int(external_seed),
            "run_index": int(run_index),
            "runtime_seconds": runtime_seconds,
            "environment_fingerprint": environment["environment_fingerprint"],
            "code_fingerprint": code_hash,
            "config_fingerprint": config_hash,
        }
        write_json(status_path, completed_status)
        return completed_status

    except Exception as error:
        failed = {
            "status": "failed",
            "dataset": dataset,
            "model": model_name,
            "group": group,
            "mode": mode,
            "external_seed": int(external_seed),
            "run_index": int(run_index),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(status_path, failed)
        raise
