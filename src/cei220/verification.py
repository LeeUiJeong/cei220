from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import read_json, result_root, write_json
from .jobs import build_jobs, expected_count_per_combo


def verify(project_root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    root = result_root(project_root, config)
    jobs = build_jobs(config)
    expected_total = len(jobs)
    expected_per_combo = expected_count_per_combo(config)

    metrics_path = root / "aggregate_metrics.csv"
    environment_path = root / "environment_by_run.csv"
    if not metrics_path.exists() or not environment_path.exists():
        raise RuntimeError("Run analysis before verification.")

    metrics = pd.read_csv(metrics_path)
    environments = pd.read_csv(environment_path)

    combo_counts = metrics.groupby(["dataset", "model"]).size().to_dict()
    combo_checks = {}
    for item in config["matrix"]:
        key = (item["dataset"], item["model"])
        actual = int(combo_counts.get(key, 0))
        combo_checks[f"{key[0]}__{key[1]}"] = {
            "expected": expected_per_combo,
            "actual": actual,
            "passed": actual == expected_per_combo,
        }

    expected_job_counts = Counter((job.dataset, job.model, job.group, job.mode) for job in jobs)
    actual_job_counts = metrics.groupby(["dataset", "model", "group", "mode"]).size().to_dict()
    protocol_checks = {}
    for key, expected in expected_job_counts.items():
        actual = int(actual_job_counts.get(key, 0))
        protocol_checks["/".join(key)] = {"expected": expected, "actual": actual, "passed": actual == expected}

    required_common = {
        "metrics.json",
        "metadata.json",
        "policy_log.jsonl",
        "model_state_only.pt",
        "checkpoint.pt",
        "hashes.json",
        "run_status.json",
    }
    missing_files: list[dict[str, str]] = []
    failed_statuses: list[dict[str, str]] = []
    for run_dir_value in metrics["run_dir"]:
        run_dir = Path(run_dir_value)
        if not run_dir.is_absolute():
            run_dir = Path(project_root).resolve() / run_dir
        mode = read_json(run_dir / "metrics.json")["mode"]
        required = set(required_common)
        if mode != "baseline":
            required.add("seed_transparency_log.jsonl")
        for filename in required:
            if not (run_dir / filename).exists():
                missing_files.append({"run_dir": str(run_dir), "file": filename})
        status = read_json(run_dir / "run_status.json") if (run_dir / "run_status.json").exists() else {}
        if status.get("status") != "completed":
            failed_statuses.append({"run_dir": str(run_dir), "status": str(status.get("status"))})

    integrity = {
        "total_runs": {"expected": expected_total, "actual": int(len(metrics)), "passed": len(metrics) == expected_total},
        "combo_counts": combo_checks,
        "protocol_counts": protocol_checks,
        "missing_files": missing_files,
        "failed_or_incomplete_runs": failed_statuses,
        "environment_fingerprint_count": int(environments["environment_fingerprint"].nunique()),
        "code_fingerprint_count": int(environments["code_fingerprint"].nunique()),
        "config_fingerprint_count": int(environments["config_fingerprint"].nunique()),
    }
    integrity_passed = (
        integrity["total_runs"]["passed"]
        and all(item["passed"] for item in combo_checks.values())
        and all(item["passed"] for item in protocol_checks.values())
        and not missing_files
        and not failed_statuses
        and integrity["environment_fingerprint_count"] == 1
        and integrity["code_fingerprint_count"] == 1
        and integrity["config_fingerprint_count"] == 1
    )

    hypothesis_checks: list[dict[str, Any]] = []
    for (dataset, model, mode), frame in metrics[metrics["group"] == "main"].groupby(["dataset", "model", "mode"]):
        unique_hashes = int(frame["model_state_sha256"].nunique())
        if mode in {"cei_lite", "cei_strict", "perturbed"}:
            hypothesis_checks.append(
                {
                    "check": "main_model_state_reproducibility",
                    "dataset": dataset,
                    "model": model,
                    "mode": mode,
                    "expected_unique_hashes": 1,
                    "actual_unique_hashes": unique_hashes,
                    "passed": unique_hashes == 1,
                }
            )

    seed_frame = metrics[metrics["group"] == "seed_generalization"]
    for (dataset, model, seed), frame in seed_frame.groupby(["dataset", "model", "external_seed"]):
        unique_hashes = int(frame["model_state_sha256"].nunique())
        hypothesis_checks.append(
            {
                "check": "seed_internal_reproducibility",
                "dataset": dataset,
                "model": model,
                "external_seed": int(seed),
                "expected_unique_hashes": 1,
                "actual_unique_hashes": unique_hashes,
                "passed": unique_hashes == 1,
            }
        )

    strong_frame = metrics[metrics["group"] == "strong_perturbation"]
    for (dataset, model, mode), frame in strong_frame.groupby(["dataset", "model", "mode"]):
        unique_hashes = int(frame["model_state_sha256"].nunique())
        hypothesis_checks.append(
            {
                "check": "strong_perturbation_changes_model_state",
                "dataset": dataset,
                "model": model,
                "mode": mode,
                "expected_minimum_unique_hashes": 2,
                "actual_unique_hashes": unique_hashes,
                "passed": unique_hashes >= 2,
            }
        )

    report = {
        "execution_integrity_passed": integrity_passed,
        "scientific_expectations_passed": all(item["passed"] for item in hypothesis_checks),
        "integrity": integrity,
        "hypothesis_checks": hypothesis_checks,
        "note": "Scientific expectation checks are reported separately from execution integrity; unexpected results remain valid research outcomes.",
    }
    write_json(root / "verification_report.json", report)
    return report
