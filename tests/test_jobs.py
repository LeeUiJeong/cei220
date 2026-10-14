from pathlib import Path

from cei220.config import load_config
from cei220.jobs import build_jobs, expected_count_per_combo


def test_full_job_count_is_220():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "experiment_220.yaml")
    jobs = build_jobs(config)
    assert expected_count_per_combo(config) == 55
    assert len(jobs) == 220
    counts = {}
    for job in jobs:
        counts[job.combo] = counts.get(job.combo, 0) + 1
    assert set(counts.values()) == {55}
