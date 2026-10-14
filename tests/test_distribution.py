"""Offline distribution checks; these are not the dissertation's 220 runs."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset

from cei220.config import load_config
from cei220.hashing import hash_state_dict

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cei_independent_audit", ROOT / "audit/audit_model_state_hashes.py")
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


@pytest.mark.parametrize("kind", ["float", "noncontiguous", "integer_buffer", "empty"])
def test_independent_hash_matches_canonical_procedure(kind):
    tensors = {
        "float": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "noncontiguous": torch.arange(12, dtype=torch.float32).reshape(3, 4).T,
        "integer_buffer": torch.tensor(7, dtype=torch.int64),
        "empty": torch.empty((0, 3)),
    }
    state = {"z": tensors[kind], "a": torch.tensor([2.0])}
    assert audit_module.independent_hash_state_dict(state) == hash_state_dict(state)


def test_audit_loads_saved_model_on_cpu(tmp_path):
    state = {"weights": torch.tensor([[0.25, 0.5]])}
    path = tmp_path / "model_state_only.pt"
    torch.save(state, path)
    loaded = audit_module.load_state_dict_cpu(path)
    assert loaded["weights"].device.type == "cpu"
    assert audit_module.independent_hash_state_dict(loaded) == hash_state_dict(state)


def test_audit_resolves_run_identity_without_old_absolute_path(tmp_path):
    row = {"dataset": "cifar10", "model": "smallcnn", "group": "main", "mode": "cei_strict", "external_seed": 20236573, "run_index": 2, "run_dir": "unavailable-old-machine-path"}
    assert audit_module.resolve_run_dir(tmp_path, row) == tmp_path / "cifar10/smallcnn/main/cei_strict/seed_20236573/run_002"


@pytest.mark.parametrize(
    ("output_root", "experiment_name"),
    [("results", "cei_220_reference_v1"), ("custom results", "renamed experiment")],
)
def test_notebook_setup_is_portable(tmp_path, monkeypatch, output_root, experiment_name):
    project = tmp_path / "relocated project"
    (project / "configs").mkdir(parents=True)
    config_path = project / "configs/experiment_220.yaml"
    config_path.write_text(
        yaml.safe_dump({"project": {"output_root": output_root, "experiment_name": experiment_name}}),
        encoding="utf-8",
    )
    (project / "audit").mkdir()
    (project / "audit/audit_model_state_hashes.py").write_text("# test placeholder\n", encoding="utf-8")
    result_root = project / output_root / experiment_name
    result_root.mkdir(parents=True)
    (result_root / "aggregate_metrics.csv").write_text("test fixture\n", encoding="utf-8")
    monkeypatch.chdir(project / "audit")
    notebook = json.loads((ROOT / "audit/CEI_220_Model_State_Rehash_Audit.ipynb").read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["outputs"] == [] and cell["execution_count"] is None for cell in cells)
    setup = next("".join(cell["source"]) for cell in cells if "def find_project_root" in "".join(cell["source"]))
    namespace = {}
    exec(compile(setup, "audit_notebook_setup", "exec"), namespace)
    assert namespace["PROJECT_ROOT"] == project
    assert namespace["SCRIPT"] == project / "audit/audit_model_state_hashes.py"
    assert namespace["RESULT_ROOT"] == result_root
    assert namespace["CONFIG_PATH"] == config_path

    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    execution = next("".join(cell["source"]) for cell in cells if "command = [" in "".join(cell["source"]))
    exec(compile(execution, "audit_notebook_command", "exec"), namespace)
    command, kwargs = calls[0]
    assert command[command.index("--config") + 1] == str(config_path)
    assert command[command.index("--result-root") + 1] == str(result_root)
    assert kwargs == {"cwd": project, "check": True}


@pytest.mark.parametrize("case", ["valid", "missing_manifest", "changed_file"])
def test_distribution_checker_reports_integrity_failures(tmp_path, case):
    project = tmp_path / "distribution fixture"
    shutil.copytree(
        ROOT, project,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "*.egg-info", ".git"),
    )
    manifest_path = project / "DISTRIBUTION_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Construct current fixture hashes so this regression test also runs before packaging.
    for record in manifest["files"]:
        content = (project / record["path"]).read_bytes()
        record["sha256"] = hashlib.sha256(content).hexdigest()
        record["bytes"] = len(content)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    if case == "missing_manifest":
        manifest_path.unlink()
    elif case == "changed_file":
        (project / "README_KO.md").write_text("deliberately changed test fixture\n", encoding="utf-8")

    outcome = subprocess.run(
        [sys.executable, "-B", str(project / "tools/check_distribution.py")],
        cwd=project,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    report = json.loads(outcome.stdout)
    assert report["manifest_present"] is (case != "missing_manifest")
    if case == "valid":
        assert outcome.returncode == 0
        assert report["file_checks_passed"] is True
        assert report["manifest_files_checked"] == len(manifest["files"])
        assert report["errors"] == []
    else:
        assert outcome.returncode != 0
        assert report["file_checks_passed"] is False
        if case == "missing_manifest":
            assert report["manifest_files_checked"] == 0
            assert "Missing distribution manifest: DISTRIBUTION_MANIFEST.json" in report["errors"]
        else:
            assert "Distribution manifest mismatch: README_KO.md" in report["errors"]


class SyntheticImages(Dataset):
    """Deterministic tiny PIL images; the original torchvision transforms still run."""
    classes = [str(i) for i in range(10)]

    def __init__(self, transform, train):
        self.transform = transform
        self.length = 8 if train else 4

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        values = (np.arange(32 * 32 * 3).reshape(32, 32, 3) + 17 * index) % 256
        image = Image.fromarray(values.astype(np.uint8))
        return self.transform(image), index % 10


def test_offline_pipeline_and_audit_success_mismatch_missing(tmp_path, monkeypatch):
    from cei220 import data
    from cei220.analysis import summarize
    from cei220.jobs import build_jobs
    from cei220.trainer import run_job
    from cei220.verification import verify

    def datasets_without_download(name, root, download=True):
        assert name == "cifar10"
        train_transform, test_transform = data.build_transforms(name)
        return SyntheticImages(train_transform, True), SyntheticImages(test_transform, False)

    monkeypatch.setattr(data, "load_datasets", datasets_without_download)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_benchmark = torch.backends.cudnn.benchmark
    old_cudnn_deterministic = torch.backends.cudnn.deterministic
    old_fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.set_num_threads(1)
    try:
        project = tmp_path / "offline project with spaces"
        (project / "configs").mkdir(parents=True)
        config = copy.deepcopy(load_config(ROOT / "configs/smoke.yaml"))
        config.pop("_config_path", None)
        config["project"]["experiment_name"] = "offline_synthetic_only"
        config["runtime"].update(device="cpu", batch_size=4)
        config_path = project / "configs/experiment_220.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        manifest = project / "data/manifests/cifar10.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"dataset_fingerprint": "synthetic-fixture-not-research-data"}), encoding="utf-8")
        jobs = build_jobs(config)
        assert len(jobs) == 7
        for job in jobs:
            result = run_job(project_root=project, config=config, dataset=job.dataset, model_name=job.model, group=job.group, mode=job.mode, external_seed=job.external_seed, run_index=job.run_index)
            assert result["status"] == "completed"
        result_root = summarize(project, config)
        report = verify(project, config)
        assert report["execution_integrity_passed"] is True
        assert report["scientific_expectations_passed"] is False
        command = [sys.executable, str(ROOT / "audit/audit_model_state_hashes.py"), "--project-root", str(project), "--expected-runs", "7", "--progress-every", "100"]
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}

        def execute_audit():
            outcome = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True, encoding="utf-8", timeout=90)
            summary = json.loads((result_root / "model_state_rehash_summary.json").read_text(encoding="utf-8"))
            return outcome, summary

        outcome, summary = execute_audit()
        assert outcome.returncode == 0, outcome.stdout + outcome.stderr
        assert summary["audit_passed_runs"] == 7 and summary["all_passed"] is True

        model_files = sorted(result_root.glob("**/model_state_only.pt"))
        original_bytes = model_files[0].read_bytes()
        state = torch.load(model_files[0], map_location="cpu", weights_only=True)
        first_key = next(iter(state))
        state[first_key] = state[first_key] + 1
        torch.save(state, model_files[0])
        outcome, summary = execute_audit()
        assert outcome.returncode != 0 and summary["audit_failed_runs"] == 1
        assert summary["all_passed"] is False

        model_files[0].write_bytes(original_bytes)
        model_files[1].rename(model_files[1].with_suffix(".held"))
        outcome, summary = execute_audit()
        assert outcome.returncode != 0 and summary["error_runs"] == 1
        assert summary["model_files_loaded"] == 6 and summary["all_passed"] is False
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)
        torch.backends.cudnn.benchmark = old_benchmark
        torch.backends.cudnn.deterministic = old_cudnn_deterministic
        torch.utils.deterministic.fill_uninitialized_memory = old_fill
