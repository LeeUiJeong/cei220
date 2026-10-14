"""Read-only checks for preserved research files and the distribution manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(root: Path, name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Invalid relative manifest path: {name}")
    return root / name


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    provenance = json.loads((root / "docs/SOURCE_PROVENANCE.json").read_text(encoding="utf-8"))
    errors = []
    preserved = 0
    core = []
    for record in provenance["files"]:
        name = record["source_path"]
        destination = record["distribution_path"]
        is_core = name.startswith(("src/", "scripts/", "configs/")) or name in {"pyproject.toml", "requirements-lock.txt"}
        must_preserve = is_core or name.startswith(("results/", "tests/")) or name in {"environment.yml", "CITATION.cff", "audit/audit_model_state_hashes.py"}
        if is_core:
            core.append(name)
        if must_preserve:
            preserved += 1
            path = safe_path(root, destination)
            if not path.is_file() or sha256(path) != record["source_sha256"]:
                errors.append(f"Changed or missing preserved file: {destination}")

    digest = hashlib.sha256()
    ordered = []
    for prefix in ["src/", "scripts/", "configs/"]:
        ordered.extend(sorted(name for name in core if name.startswith(prefix)))
    ordered.extend(["pyproject.toml", "requirements-lock.txt"])
    for name in ordered:
        path = root / name
        if path.is_file():
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
    source_fingerprint = digest.hexdigest()
    if source_fingerprint != provenance["recorded_experiment_code_fingerprint"]:
        errors.append("Recorded source-file-set fingerprint differs")

    manifest_path = root / "DISTRIBUTION_MANIFEST.json"
    manifest_count = 0
    manifest_present = manifest_path.is_file()
    if not manifest_present:
        errors.append("Missing distribution manifest: DISTRIBUTION_MANIFEST.json")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest["files"]:
            manifest_count += 1
            path = safe_path(root, record["path"])
            if not path.is_file() or sha256(path) != record["sha256"]:
                errors.append(f"Distribution manifest mismatch: {record['path']}")

    report = {
        "preserved_files_checked": preserved,
        "recorded_core_files_checked": len(ordered),
        "recorded_source_fingerprint": source_fingerprint,
        "manifest_present": manifest_present,
        "manifest_files_checked": manifest_count,
        "file_checks_passed": not errors,
        "license_file_present": (root / "LICENSE").is_file(),
        "publication_note": "License and author publication decisions remain separate from these file-integrity checks.",
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
