# CEI 220-Run Reference Implementation

Controlled Entropy Injection (CEI) reference implementation for repeated deep-learning experiments, seed-path tracing, and model-state hash auditing.

**Project code and accompanying software documentation are licensed under the MIT License.** Source repository: [LeeUiJeong/cei220](https://github.com/LeeUiJeong/cei220). See the [publication checklist](docs/PUBLICATION_CHECKLIST.md). 한국어: [README_KO.md](README_KO.md).

## Included implementation and evidence

The original training, seed, policy, configuration, aggregation, verification, and independent-audit code is preserved. The historical protocol combines CIFAR-10 and Fashion-MNIST with SmallCNN and ResNet18-small: four combinations, 55 runs each, 220 runs total. Each combination contains 40 basic runs, 9 within-external-seed repeats, and 6 strong seed-perturbation runs.

`perturbed` is logging-only perturbation. `perturbed_model_seed` and `perturbed_loader_seed` change applied seeds. The existing CSV/JSON summaries are under `reference_results/cei_220_reference_v1/`; new runs write under `results/`.

| Purpose | Entry point |
| --- | --- |
| Read historical results without training | [Reference results](reference_results/README.md) |
| Check functionality without datasets or GPU computation | CPU installation and offline tests below |
| Run a smaller real-data experiment | Smoke workflow below |
| Run all 220 jobs | Full workflow below |
| Re-audit stored models | [Audit instructions](audit/README.md); complete model files required |

## Installation

Use a dedicated **Python 3.10** environment. Run commands from the project root. Package metadata remains at its original `1.0.0`; deployment-only changes are recorded in [RELEASE_NOTES.md](RELEASE_NOTES.md).

### CPU functional checks

After creating and activating a Python 3.10 environment:

```text
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.2.5 pandas==2.3.3 PyYAML==6.0.2 pytest==9.0.2
python -m pip install . --no-deps
python -m pytest -q
python tools/check_distribution.py
python scripts/run_all.py --config configs/experiment_220.yaml --dry-run
```

The dry run prints 220 planned jobs without downloading data or training. Offline tests use tiny synthetic images and temporary directories. Passing these checks does not establish reproduction of the historical GPU results. CPU packages use the official [PyTorch wheel index](https://download.pytorch.org/whl/cpu/torch/).

### Dissertation-target environment

```text
conda env create -f environment.yml
conda activate cei220
python -m pip install . --no-deps
python scripts/check_environment.py --config configs/experiment_220.yaml
```

Recorded experiment environment: Python 3.10.20, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, NumPy 2.2.5, CUDA build 12.8, cuDNN 91900, NVIDIA GeForce RTX 4090. The checker tests the configured version prefixes, CUDA/cuDNN values, and GPU name, not every environment property. Driver, OS, and hardware differences still matter. Original environment files are retained rather than silently upgrading research dependencies.

## Tests and real-data smoke workflow

`python -m pytest -q` includes the six original unit tests and additional distribution tests. It does not download datasets. See [validation scope](docs/VALIDATION.md).

The smoke workflow below **downloads CIFAR-10** and performs **seven one-epoch runs** on the real dataset:

```text
python scripts/check_environment.py --config configs/smoke.yaml --allow-mismatch
python scripts/prepare_data.py --config configs/smoke.yaml
python scripts/run_all.py --config configs/smoke.yaml
python scripts/analyze.py --config configs/smoke.yaml
```

The smoke configuration has only one run per strong-perturbation type, so those groups cannot satisfy the full scientific expectation of at least two distinct hashes. Do not claim full scientific-expectation verification from this smoke run.

## Full experiment

This downloads datasets and performs 220 training/evaluation runs:

```text
python scripts/run_pipeline.py --config configs/experiment_220.yaml
```

The pipeline checks the environment, prepares data, executes independent processes, aggregates results, and runs automatic verification. Unit tests and the independent rehash audit are separate procedures.

Re-running without `--overwrite` skips completed runs; this does not restore interrupted training state. **`--overwrite` deletes existing per-run outputs.** Use a new experiment name or a working copy when preserving earlier runs.

For one combination:

```text
python scripts/run_all.py --config configs/experiment_220.yaml --combo fashion_mnist__resnet18_small
```

The full-protocol verifier expects the configured complete matrix, so partial runs should not be reported as the complete 220-run experiment.

## Results and independent audit

The distribution does **not** include per-run model files, checkpoints, all detailed logs, or dataset files. No access URL for the full original evidence set has been supplied. The maintainer must arrange separate access before promising independent re-auditing of the original 220 stored models.

After restoring a **working copy** of the complete result tree to the configured location:

```text
python audit/audit_model_state_hashes.py --project-root . --expected-runs 220
```

This loads stored model states on the CPU without retraining. It overwrites four audit reports in the selected result directory. See [audit instructions](audit/README.md) for external result directories and notebook use.

## Interpretation and provenance

- Execution integrity and scientific expectations are recorded separately.
- Same-seed repetition and deliberately different perturbation seeds are different comparison units.
- File hashes and model-state content hashes answer different questions.
- Hash agreement does not establish the correctness of every operation or guarantee equality across environments.
- The original core code, configuration, and reference results are preserved. Documentation and notebook path setup differ; see [source provenance](docs/SOURCE_PROVENANCE.json) and [release notes](RELEASE_NOTES.md).
- Historical absolute paths retained in reference records are provenance fields, not executable paths on a reader's machine.

## Citation and license

The author name is **Uijeong Lee**. [CITATION.cff](CITATION.cff) identifies the software version `1.0.0` and this repository. The historical citation file is preserved unchanged as [docs/ORIGINAL_CITATION.cff](docs/ORIGINAL_CITATION.cff). No release tag, DOI, ORCID, or final dissertation citation is asserted by this upload copy. The preparation date in distribution records is not a GitHub release date.

The project code and accompanying software documentation are licensed under the [MIT License](LICENSE). You may use, modify, and redistribute the software, including commercially, subject to retaining the copyright and permission notices. The software is provided without warranty under the license terms.

This software license does not assign new licensing terms to the historical research data and reports in `reference_results/`, the dissertation, or third-party libraries and datasets. See [license scope](LICENSE_STATUS.md). The citation request is separate from the MIT license conditions.
