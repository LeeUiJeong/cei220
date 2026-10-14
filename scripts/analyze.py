from __future__ import annotations

import argparse

from _bootstrap import PROJECT_ROOT
from cei220.analysis import summarize
from cei220.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiment_220.yaml"))
    args = parser.parse_args()
    root = summarize(PROJECT_ROOT, load_config(args.config))
    print(f"Analysis written to: {root}")


if __name__ == "__main__":
    main()
