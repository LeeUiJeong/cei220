from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute model-state hashes from all model_state_only.pt files "
            "and compare them with the hashes recorded during the CEI 220-run experiment."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root containing configs/experiment_220.yaml. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Experiment YAML. Defaults to configs/experiment_220.yaml.",
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=None,
        help="Result root. Defaults to the path declared by the YAML config.",
    )
    parser.add_argument(
        "--expected-runs",
        type=int,
        default=220,
        help="Expected number of runs. Default: 220.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N runs. Default: 10.",
    )
    return parser.parse_args()


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    candidates = [start, *start.parents]
    for candidate in candidates:
        if (candidate / "configs" / "experiment_220.yaml").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate a project root containing configs/experiment_220.yaml. "
        "Run this script inside the CEI_220_Reference_Implementation project or "
        "pass --project-root explicitly."
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def independent_hash_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> str:
    """
    Standalone reimplementation of the canonical CEI model-state hash.

    This intentionally does not import src/cei220/hashing.py, so the audit
    independently recomputes the digest from the saved tensor contents.
    """
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"State-dict entry {key!r} is {type(value).__name__}, not torch.Tensor."
            )
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_state_dict_cpu(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Compatibility fallback for older PyTorch releases.
        payload = torch.load(path, map_location="cpu")

    if isinstance(payload, (dict, OrderedDict)):
        if "model_state_dict" in payload and isinstance(
            payload["model_state_dict"], (dict, OrderedDict)
        ):
            return payload["model_state_dict"]
        return payload

    raise TypeError(
        f"{path} did not contain a state_dict-compatible mapping; "
        f"loaded type: {type(payload).__name__}"
    )


def resolve_run_dir(
    result_root: Path,
    row: pd.Series,
) -> Path:
    # Reconstruct from the stable artifact schema rather than relying on the
    # absolute Windows path stored in aggregate_metrics.csv.
    return (
        result_root
        / str(row["dataset"])
        / str(row["model"])
        / str(row["group"])
        / str(row["mode"])
        / f"seed_{int(row['external_seed'])}"
        / f"run_{int(row['run_index']):03d}"
    )


def bool_all(values: list[bool]) -> bool:
    return bool(values) and all(values)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else find_project_root(Path.cwd())
    )
    config_path = (
        args.config.resolve()
        if args.config is not None
        else project_root / "configs" / "experiment_220.yaml"
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.result_root is not None:
        result_root = args.result_root.resolve()
    else:
        result_root = (
            project_root
            / str(config["project"]["output_root"])
            / str(config["project"]["experiment_name"])
        ).resolve()

    aggregate_path = result_root / "aggregate_metrics.csv"
    if not aggregate_path.exists():
        raise FileNotFoundError(
            f"aggregate_metrics.csv not found: {aggregate_path}"
        )

    aggregate = pd.read_csv(aggregate_path)
    required_columns = {
        "dataset",
        "model",
        "group",
        "mode",
        "external_seed",
        "run_index",
        "model_state_sha256",
    }
    missing_columns = sorted(required_columns - set(aggregate.columns))
    if missing_columns:
        raise ValueError(
            f"aggregate_metrics.csv is missing required columns: {missing_columns}"
        )

    aggregate = aggregate.sort_values(
        ["dataset", "model", "group", "mode", "external_seed", "run_index"]
    ).reset_index(drop=True)

    print("=== CEI model-state independent rehash audit ===")
    print("Project root :", project_root)
    print("Result root  :", result_root)
    print("Runs in CSV  :", len(aggregate))
    print("PyTorch      :", torch.__version__)
    print("Python       :", platform.python_version())
    print()

    audit_rows: list[dict[str, Any]] = []

    for index, row in aggregate.iterrows():
        run_dir = resolve_run_dir(result_root, row)
        state_path = run_dir / "model_state_only.pt"
        hashes_path = run_dir / "hashes.json"
        metrics_path = run_dir / "metrics.json"

        record: dict[str, Any] = {
            "dataset": str(row["dataset"]),
            "model": str(row["model"]),
            "group": str(row["group"]),
            "mode": str(row["mode"]),
            "external_seed": int(row["external_seed"]),
            "run_index": int(row["run_index"]),
            "run_dir": str(run_dir),
            "model_state_file": str(state_path),
            "model_state_file_exists": state_path.exists(),
            "recorded_aggregate_hash": str(row["model_state_sha256"]),
            "recorded_metrics_hash": None,
            "recorded_hashes_json_hash": None,
            "recomputed_state_hash": None,
            "aggregate_match": False,
            "metrics_match": False,
            "hashes_json_match": False,
            "all_recorded_state_hashes_agree": False,
            "recorded_model_file_sha256": None,
            "recomputed_model_file_sha256": None,
            "model_file_sha256_match": False,
            "load_status": "not_run",
            "error": "",
        }

        try:
            if not state_path.exists():
                raise FileNotFoundError(f"Missing model state file: {state_path}")
            if not hashes_path.exists():
                raise FileNotFoundError(f"Missing hashes.json: {hashes_path}")
            if not metrics_path.exists():
                raise FileNotFoundError(f"Missing metrics.json: {metrics_path}")

            hashes = read_json(hashes_path)
            metrics = read_json(metrics_path)

            recorded_metrics_hash = str(metrics.get("model_state_sha256", ""))
            recorded_hashes_json_hash = str(
                hashes.get("__model_state_dict__", "")
            )
            recorded_model_file_sha256 = str(
                hashes.get("model_state_only.pt", "")
            )

            state_dict = load_state_dict_cpu(state_path)
            recomputed_state_hash = independent_hash_state_dict(state_dict)
            recomputed_model_file_sha256 = sha256_file(state_path)

            recorded_state_hashes = [
                str(row["model_state_sha256"]),
                recorded_metrics_hash,
                recorded_hashes_json_hash,
            ]

            record.update(
                {
                    "recorded_metrics_hash": recorded_metrics_hash,
                    "recorded_hashes_json_hash": recorded_hashes_json_hash,
                    "recomputed_state_hash": recomputed_state_hash,
                    "aggregate_match": (
                        recomputed_state_hash
                        == str(row["model_state_sha256"])
                    ),
                    "metrics_match": (
                        recomputed_state_hash == recorded_metrics_hash
                    ),
                    "hashes_json_match": (
                        recomputed_state_hash == recorded_hashes_json_hash
                    ),
                    "all_recorded_state_hashes_agree": (
                        len(set(recorded_state_hashes)) == 1
                        and all(recorded_state_hashes)
                    ),
                    "recorded_model_file_sha256": recorded_model_file_sha256,
                    "recomputed_model_file_sha256": (
                        recomputed_model_file_sha256
                    ),
                    "model_file_sha256_match": (
                        recomputed_model_file_sha256
                        == recorded_model_file_sha256
                    ),
                    "load_status": "ok",
                }
            )

            del state_dict
            gc.collect()

        except Exception as exc:
            record["load_status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"

        audit_rows.append(record)

        done = index + 1
        if done % max(1, args.progress_every) == 0 or done == len(aggregate):
            elapsed = time.perf_counter() - started
            print(
                f"[{done:03d}/{len(aggregate):03d}] "
                f"elapsed={elapsed / 60:.1f} min"
            )

    audit = pd.DataFrame(audit_rows)

    match_columns = [
        "aggregate_match",
        "metrics_match",
        "hashes_json_match",
        "all_recorded_state_hashes_agree",
        "model_file_sha256_match",
    ]
    audit["state_rehash_all_match"] = audit[
        ["aggregate_match", "metrics_match", "hashes_json_match"]
    ].all(axis=1)
    audit["audit_passed"] = (
        (audit["load_status"] == "ok")
        & audit["state_rehash_all_match"]
        & audit["all_recorded_state_hashes_agree"]
        & audit["model_file_sha256_match"]
    )

    audit_path = result_root / "model_state_rehash_audit.csv"
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")

    mismatches = audit[~audit["audit_passed"]].copy()
    mismatch_path = result_root / "model_state_rehash_mismatches.csv"
    mismatches.to_csv(mismatch_path, index=False, encoding="utf-8-sig")

    group_keys = [
        "dataset",
        "model",
        "group",
        "mode",
        "external_seed",
    ]
    group_summary = (
        audit.groupby(group_keys, dropna=False)
        .agg(
            runs=("run_index", "count"),
            loaded_ok=("load_status", lambda x: int((x == "ok").sum())),
            audit_passed=("audit_passed", "sum"),
            recomputed_unique_state_hashes=(
                "recomputed_state_hash",
                lambda x: int(pd.Series(x).dropna().nunique()),
            ),
            recorded_unique_state_hashes=(
                "recorded_aggregate_hash",
                lambda x: int(pd.Series(x).dropna().nunique()),
            ),
            file_sha256_matches=("model_file_sha256_match", "sum"),
        )
        .reset_index()
    )
    group_summary["all_runs_passed"] = (
        group_summary["audit_passed"] == group_summary["runs"]
    )
    group_summary["unique_hash_count_agrees"] = (
        group_summary["recomputed_unique_state_hashes"]
        == group_summary["recorded_unique_state_hashes"]
    )

    group_summary_path = result_root / "model_state_rehash_group_summary.csv"
    group_summary.to_csv(
        group_summary_path, index=False, encoding="utf-8-sig"
    )

    elapsed_seconds = time.perf_counter() - started
    summary = {
        "audit_name": "CEI 220 model-state independent rehash audit",
        "project_root": str(project_root),
        "result_root": str(result_root),
        "aggregate_metrics": str(aggregate_path),
        "expected_runs": int(args.expected_runs),
        "rows_in_aggregate_metrics": int(len(aggregate)),
        "model_files_found": int(audit["model_state_file_exists"].sum()),
        "model_files_loaded": int((audit["load_status"] == "ok").sum()),
        "audit_passed_runs": int(audit["audit_passed"].sum()),
        "audit_failed_runs": int((~audit["audit_passed"]).sum()),
        "aggregate_hash_matches": int(audit["aggregate_match"].sum()),
        "metrics_hash_matches": int(audit["metrics_match"].sum()),
        "hashes_json_matches": int(audit["hashes_json_match"].sum()),
        "file_sha256_matches": int(audit["model_file_sha256_match"].sum()),
        "recorded_state_hash_sources_agree": int(
            audit["all_recorded_state_hashes_agree"].sum()
        ),
        "error_runs": int((audit["load_status"] == "error").sum()),
        "group_count": int(len(group_summary)),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": elapsed_seconds / 60.0,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "all_passed": bool(
            len(aggregate) == int(args.expected_runs)
            and len(audit) == int(args.expected_runs)
            and audit["audit_passed"].all()
            and group_summary["all_runs_passed"].all()
            and group_summary["unique_hash_count_agrees"].all()
        ),
        "outputs": {
            "run_audit_csv": str(audit_path),
            "mismatch_csv": str(mismatch_path),
            "group_summary_csv": str(group_summary_path),
        },
    }

    summary_path = result_root / "model_state_rehash_summary.json"
    summary["outputs"]["summary_json"] = str(summary_path)
    write_json(summary_path, summary)

    print("\n=== Audit summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nGenerated files:")
    print("-", audit_path)
    print("-", mismatch_path)
    print("-", group_summary_path)
    print("-", summary_path)

    if not summary["all_passed"]:
        raise SystemExit(
            "The rehash audit found at least one mismatch or missing file. "
            "Review model_state_rehash_mismatches.csv."
        )


if __name__ == "__main__":
    main()
