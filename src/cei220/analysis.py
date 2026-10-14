from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import read_json, result_root, write_json


def collect_results(project_root: str | Path, config: dict[str, Any]):
    root = result_root(project_root, config)
    metric_rows: list[dict[str, Any]] = []
    hash_rows: list[dict[str, Any]] = []
    environment_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []

    for metrics_path in sorted(root.glob("**/metrics.json")):
        run_dir = metrics_path.parent
        metrics = read_json(metrics_path)
        metadata = read_json(run_dir / "metadata.json")
        hashes = read_json(run_dir / "hashes.json")
        environment = metadata["environment"]

        metric_rows.append({**metrics, "run_dir": str(run_dir)})
        for artifact, sha256 in hashes.items():
            hash_rows.append(
                {
                    "dataset": metrics["dataset"],
                    "model": metrics["model"],
                    "group": metrics["group"],
                    "mode": metrics["mode"],
                    "external_seed": metrics["external_seed"],
                    "run_index": metrics["run_index"],
                    "artifact": artifact,
                    "sha256": sha256,
                }
            )
        environment_rows.append(
            {
                "dataset": metrics["dataset"],
                "model": metrics["model"],
                "group": metrics["group"],
                "mode": metrics["mode"],
                "external_seed": metrics["external_seed"],
                "run_index": metrics["run_index"],
                "environment_fingerprint": environment["environment_fingerprint"],
                "python": environment["python"],
                "torch": environment["torch"],
                "torchvision": environment["torchvision"],
                "numpy": environment["numpy"],
                "cuda_build": environment["cuda_build"],
                "cudnn": environment["cudnn"],
                "gpu_name": environment["gpu_name"],
                "nvidia_driver": environment["nvidia_driver"],
                "code_fingerprint": metadata["code_fingerprint"],
                "config_fingerprint": metadata["config_fingerprint"],
                "dataset_fingerprint": metadata["dataset_fingerprint"],
            }
        )

        policy_path = run_dir / "policy_log.jsonl"
        if policy_path.exists():
            for line in policy_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = __import__("json").loads(line)
                policy_rows.append(
                    {
                        "dataset": metrics["dataset"],
                        "model": metrics["model"],
                        "group": metrics["group"],
                        "mode": metrics["mode"],
                        "external_seed": metrics["external_seed"],
                        "run_index": metrics["run_index"],
                        "event": event.get("event"),
                        "level": event.get("level"),
                        "action": event.get("action"),
                    }
                )

    metrics_df = pd.DataFrame(metric_rows)
    hashes_df = pd.DataFrame(hash_rows)
    environments_df = pd.DataFrame(environment_rows)
    policies_df = pd.DataFrame(policy_rows)
    return root, metrics_df, hashes_df, environments_df, policies_df


def summarize(project_root: str | Path, config: dict[str, Any]) -> Path:
    root, metrics, hashes, environments, policies = collect_results(project_root, config)
    if metrics.empty:
        raise RuntimeError(f"No completed metrics found under {root}")

    sort_cols = ["dataset", "model", "group", "mode", "external_seed", "run_index"]
    metrics = metrics.sort_values(sort_cols)
    hashes = hashes.sort_values(sort_cols + ["artifact"])
    environments = environments.sort_values(sort_cols)

    metrics.to_csv(root / "aggregate_metrics.csv", index=False)
    hashes.to_csv(root / "artifact_hashes.csv", index=False)
    environments.to_csv(root / "environment_by_run.csv", index=False)
    policies.to_csv(root / "policy_events.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    group_cols = ["dataset", "model", "group", "mode"]
    for keys, frame in metrics.groupby(group_cols):
        dataset, model, group, mode = keys
        unique_hashes = int(frame["model_state_sha256"].nunique())
        summary_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "group": group,
                "mode": mode,
                "runs": int(len(frame)),
                "external_seed_count": int(frame["external_seed"].nunique()),
                "accuracy_mean": float(frame["final_test_accuracy"].mean()),
                "accuracy_std": float(frame["final_test_accuracy"].std(ddof=0)),
                "loss_mean": float(frame["final_test_loss"].mean()),
                "loss_std": float(frame["final_test_loss"].std(ddof=0)),
                "runtime_mean_seconds": float(frame["runtime_seconds"].mean()),
                "runtime_std_seconds": float(frame["runtime_seconds"].std(ddof=0)),
                "model_state_unique_hashes": unique_hashes,
                "model_state_all_equal": unique_hashes == 1,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(group_cols)
    summary.to_csv(root / "summary_by_condition.csv", index=False)

    seed_rows: list[dict[str, Any]] = []
    seed_data = metrics[metrics["group"] == "seed_generalization"]
    for keys, frame in seed_data.groupby(["dataset", "model", "external_seed"]):
        dataset, model, external_seed = keys
        unique_hashes = int(frame["model_state_sha256"].nunique())
        seed_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "external_seed": int(external_seed),
                "runs": int(len(frame)),
                "accuracy_mean": float(frame["final_test_accuracy"].mean()),
                "accuracy_std": float(frame["final_test_accuracy"].std(ddof=0)),
                "loss_mean": float(frame["final_test_loss"].mean()),
                "loss_std": float(frame["final_test_loss"].std(ddof=0)),
                "model_state_unique_hashes": unique_hashes,
                "model_state_all_equal": unique_hashes == 1,
            }
        )
    seed_summary = pd.DataFrame(seed_rows).sort_values(["dataset", "model", "external_seed"])
    seed_summary.to_csv(root / "seed_generalization_by_seed.csv", index=False)

    hash_summary = (
        hashes.groupby(["dataset", "model", "group", "mode", "artifact"])["sha256"]
        .nunique()
        .reset_index(name="unique_hashes")
    )
    hash_summary["all_equal"] = hash_summary["unique_hashes"] == 1
    hash_summary.to_csv(root / "hash_summary_by_artifact.csv", index=False)

    environment_summary = (
        environments.groupby(
            [
                "environment_fingerprint",
                "python",
                "torch",
                "torchvision",
                "numpy",
                "cuda_build",
                "cudnn",
                "gpu_name",
                "nvidia_driver",
                "code_fingerprint",
                "config_fingerprint",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="run_count")
    )
    environment_summary.to_csv(root / "environment_summary.csv", index=False)

    main = summary[summary["group"] == "main"].copy()
    main["정확도"] = main.apply(lambda row: f"{row.accuracy_mean:.4f}±{row.accuracy_std:.4f}", axis=1)
    main["손실"] = main.apply(lambda row: f"{row.loss_mean:.4f}±{row.loss_std:.4f}", axis=1)
    main["모델 상태 해시"] = main.apply(
        lambda row: f"{int(row.model_state_unique_hashes)}({'일치' if row.model_state_all_equal else '불일치'})",
        axis=1,
    )
    main.rename(
        columns={
            "dataset": "데이터셋",
            "model": "모델",
            "mode": "조건",
            "runs": "실행 수",
            "runtime_mean_seconds": "평균 실행 시간(초)",
        },
        inplace=True,
    )
    main[
        ["데이터셋", "모델", "조건", "실행 수", "정확도", "손실", "평균 실행 시간(초)", "모델 상태 해시"]
    ].to_csv(root / "table_main_kr.csv", index=False)

    seed_table = seed_summary.copy()
    seed_table["정확도"] = seed_table.apply(lambda row: f"{row.accuracy_mean:.4f}±{row.accuracy_std:.4f}", axis=1)
    seed_table["손실"] = seed_table.apply(lambda row: f"{row.loss_mean:.4f}±{row.loss_std:.4f}", axis=1)
    seed_table["모델 상태 해시"] = seed_table.apply(
        lambda row: f"{int(row.model_state_unique_hashes)}({'일치' if row.model_state_all_equal else '불일치'})",
        axis=1,
    )
    seed_table.rename(
        columns={"dataset": "데이터셋", "model": "모델", "external_seed": "외부 시드", "runs": "실행 수"},
        inplace=True,
    )
    seed_table[["데이터셋", "모델", "외부 시드", "실행 수", "정확도", "손실", "모델 상태 해시"]].to_csv(
        root / "table_seed_generalization_kr.csv", index=False
    )

    strong = summary[summary["group"] == "strong_perturbation"].copy()
    strong["정확도"] = strong.apply(lambda row: f"{row.accuracy_mean:.4f}±{row.accuracy_std:.4f}", axis=1)
    strong["손실"] = strong.apply(lambda row: f"{row.loss_mean:.4f}±{row.loss_std:.4f}", axis=1)
    strong["모델 상태 해시"] = strong.apply(
        lambda row: f"{int(row.model_state_unique_hashes)}({'일치' if row.model_state_all_equal else '불일치'})",
        axis=1,
    )
    strong.rename(columns={"dataset": "데이터셋", "model": "모델", "mode": "조건", "runs": "실행 수"}, inplace=True)
    strong[["데이터셋", "모델", "조건", "실행 수", "정확도", "손실", "모델 상태 해시"]].to_csv(
        root / "table_strong_perturbation_kr.csv", index=False
    )

    write_json(
        root / "analysis_summary.json",
        {
            "total_completed_runs": int(len(metrics)),
            "environment_fingerprint_count": int(environments["environment_fingerprint"].nunique()),
            "code_fingerprint_count": int(environments["code_fingerprint"].nunique()),
            "config_fingerprint_count": int(environments["config_fingerprint"].nunique()),
            "result_root": str(root),
        },
    )
    return root
