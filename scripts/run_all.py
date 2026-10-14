from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from cei220.artifacts import read_json, result_root, run_directory, write_json
from cei220.config import load_config
from cei220.jobs import build_jobs, filter_jobs


def completed(run_dir: Path) -> bool:
    status_path = run_dir / "run_status.json"
    if not status_path.exists():
        return False
    try:
        return read_json(status_path).get("status") == "completed"
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiment_220.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--combo", help="Filter: e.g. cifar10__smallcnn")
    parser.add_argument("--group")
    parser.add_argument("--mode")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    jobs = filter_jobs(build_jobs(config), combo=args.combo, group=args.group, mode=args.mode)
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]

    if args.dry_run:
        for index, job in enumerate(jobs, 1):
            print(f"[{index:03d}/{len(jobs):03d}] {job.key}")
        return

    root = result_root(PROJECT_ROOT, config)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "run_manifest.json"
    manifest = {
        "experiment_name": config["project"]["experiment_name"],
        "expected_jobs_in_selection": len(jobs),
        "started_at_unix": time.time(),
        "jobs": [],
    }
    write_json(manifest_path, manifest)

    runner = PROJECT_ROOT / "scripts" / "run_one.py"
    for index, job in enumerate(jobs, 1):
        run_dir = run_directory(
            PROJECT_ROOT,
            config,
            dataset=job.dataset,
            model=job.model,
            group=job.group,
            mode=job.mode,
            external_seed=job.external_seed,
            run_index=job.run_index,
        )
        if not args.overwrite and not args.no_resume and completed(run_dir):
            print(f"[{index:03d}/{len(jobs):03d}] SKIP completed: {job.key}", flush=True)
            manifest["jobs"].append({**job.to_dict(), "status": "skipped_completed"})
            write_json(manifest_path, manifest)
            continue

        if args.overwrite and run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        command = [
            sys.executable,
            str(runner),
            "--config",
            str(Path(args.config).resolve()),
            "--dataset",
            job.dataset,
            "--model",
            job.model,
            "--group",
            job.group,
            "--mode",
            job.mode,
            "--external-seed",
            str(job.external_seed),
            "--run-index",
            str(job.run_index),
        ]
        print(f"[{index:03d}/{len(jobs):03d}] RUN {job.key}", flush=True)
        started = time.time()
        env = os.environ.copy()
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
        record = {
            **job.to_dict(),
            "status": "completed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "elapsed_seconds": time.time() - started,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        manifest["jobs"].append(record)
        write_json(manifest_path, manifest)
        if result.returncode != 0:
            manifest["status"] = "failed"
            manifest["failed_job"] = job.to_dict()
            manifest["finished_at_unix"] = time.time()
            write_json(manifest_path, manifest)
            raise SystemExit(f"Run failed: {job.key}. See {stderr_path}")

    manifest["status"] = "completed"
    manifest["finished_at_unix"] = time.time()
    write_json(manifest_path, manifest)
    print(f"Completed selection: {len(jobs)} jobs")


if __name__ == "__main__":
    main()
