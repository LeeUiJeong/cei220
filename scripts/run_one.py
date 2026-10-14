from __future__ import annotations

# This script deliberately configures process-level CUDA determinism before
# importing torch through the CEI package.
import argparse
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--external-seed", required=True, type=int)
    parser.add_argument("--run-index", required=True, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    strict_modes = {"cei_strict", "perturbed", "perturbed_loader_seed", "perturbed_model_seed"}
    if args.mode in strict_modes:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = raw_config["runtime"]["cublas_workspace_config"]
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

    from cei220.config import load_config
    from cei220.trainer import run_job

    config = load_config(args.config)
    result = run_job(
        project_root=PROJECT_ROOT,
        config=config,
        dataset=args.dataset,
        model_name=args.model,
        group=args.group,
        mode=args.mode,
        external_seed=args.external_seed,
        run_index=args.run_index,
        overwrite=args.overwrite,
    )
    import json

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
