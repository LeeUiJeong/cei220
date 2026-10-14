from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from cei220.config import load_config
from cei220.data import create_dataset_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiment_220.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    data_root = PROJECT_ROOT / config["project"]["data_root"]
    manifest_dir = data_root / "manifests"
    datasets = sorted({item["dataset"] for item in config["matrix"]})
    manifests = []
    for name in datasets:
        print(f"Preparing dataset: {name}", flush=True)
        manifests.append(create_dataset_manifest(name, data_root, manifest_dir / f"{name}.json"))
    print(json.dumps(manifests, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
