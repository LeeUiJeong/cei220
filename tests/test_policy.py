from pathlib import Path

import pytest

from cei220.policy import PolicyLogger, PolicyProfile


def test_block_policy_raises(tmp_path: Path):
    profile = PolicyProfile(
        name="test_block",
        deterministic_algorithms=True,
        cudnn_benchmark=False,
        cudnn_deterministic=True,
        fill_uninitialized_memory=True,
        cublas_workspace_config=":4096:8",
        on_violation="block",
    )
    logger = PolicyLogger(tmp_path / "policy.jsonl", profile)
    with pytest.raises(RuntimeError):
        logger.violation("test_violation")
    assert (tmp_path / "policy.jsonl").exists()
