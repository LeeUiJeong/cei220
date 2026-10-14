from __future__ import annotations

import argparse
import subprocess
import sys

from _bootstrap import PROJECT_ROOT


def call(script: str, config: str, extra: list[str] | None = None) -> None:
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / script), "--config", config]
    if extra:
        command.extend(extra)
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiment_220.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-environment-mismatch", action="store_true")
    args = parser.parse_args()

    env_extra = ["--allow-mismatch"] if args.allow_environment_mismatch else None
    call("check_environment.py", args.config, env_extra)
    call("prepare_data.py", args.config)
    run_extra = ["--overwrite"] if args.overwrite else None
    call("run_all.py", args.config, run_extra)
    call("analyze.py", args.config)
    call("verify.py", args.config)


if __name__ == "__main__":
    main()
