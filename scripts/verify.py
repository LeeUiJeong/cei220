from __future__ import annotations

import argparse
import json

from _bootstrap import PROJECT_ROOT
from cei220.config import load_config
from cei220.verification import verify


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiment_220.yaml"))
    args = parser.parse_args()
    report = verify(PROJECT_ROOT, load_config(args.config))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["execution_integrity_passed"]:
        raise SystemExit("Execution integrity verification failed.")


if __name__ == "__main__":
    main()
