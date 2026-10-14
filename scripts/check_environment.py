from __future__ import annotations

import argparse
import json

from _bootstrap import PROJECT_ROOT  # noqa: F401
from cei220.config import load_config
from cei220.environment import check_expected_environment, collect_environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "experiment_220.yaml"))
    parser.add_argument("--allow-mismatch", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    actual = collect_environment()
    result = check_expected_environment(actual, config.get("expected_environment", {}))
    print(json.dumps({"actual": actual, "validation": result}, ensure_ascii=False, indent=2))
    if not result["passed"] and not args.allow_mismatch:
        raise SystemExit("Environment check failed.")


if __name__ == "__main__":
    main()
