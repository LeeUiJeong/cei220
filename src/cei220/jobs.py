from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Job:
    dataset: str
    model: str
    group: str
    mode: str
    external_seed: int
    run_index: int

    @property
    def combo(self) -> str:
        return f"{self.dataset}__{self.model}"

    @property
    def key(self) -> str:
        return (
            f"{self.dataset}/{self.model}/{self.group}/{self.mode}/"
            f"seed_{self.external_seed}/run_{self.run_index:03d}"
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_jobs(config: dict[str, Any]) -> list[Job]:
    runtime = config["runtime"]
    protocol = config["protocol"]
    jobs: list[Job] = []

    for item in config["matrix"]:
        dataset = item["dataset"]
        model = item["model"]

        for mode in protocol["main_modes"]:
            for run_index in range(int(runtime["main_runs"])):
                jobs.append(
                    Job(
                        dataset=dataset,
                        model=model,
                        group="main",
                        mode=mode,
                        external_seed=int(runtime["external_seed"]),
                        run_index=run_index,
                    )
                )

        for seed in runtime["seed_generalization_seeds"]:
            for run_index in range(int(runtime["seed_runs_per_seed"])):
                jobs.append(
                    Job(
                        dataset=dataset,
                        model=model,
                        group="seed_generalization",
                        mode=protocol["seed_generalization_mode"],
                        external_seed=int(seed),
                        run_index=run_index,
                    )
                )

        for mode in protocol["strong_perturbation_modes"]:
            for run_index in range(int(runtime["perturbation_runs"])):
                jobs.append(
                    Job(
                        dataset=dataset,
                        model=model,
                        group="strong_perturbation",
                        mode=mode,
                        external_seed=int(runtime["external_seed"]),
                        run_index=run_index,
                    )
                )

    return jobs


def expected_count_per_combo(config: dict[str, Any]) -> int:
    runtime = config["runtime"]
    protocol = config["protocol"]
    return (
        len(protocol["main_modes"]) * int(runtime["main_runs"])
        + len(runtime["seed_generalization_seeds"]) * int(runtime["seed_runs_per_seed"])
        + len(protocol["strong_perturbation_modes"]) * int(runtime["perturbation_runs"])
    )


def filter_jobs(
    jobs: Iterable[Job],
    *,
    combo: str | None = None,
    group: str | None = None,
    mode: str | None = None,
) -> list[Job]:
    output: list[Job] = []
    for job in jobs:
        if combo and job.combo != combo:
            continue
        if group and job.group != group:
            continue
        if mode and job.mode != mode:
            continue
        output.append(job)
    return output
