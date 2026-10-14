# Local distribution validation — 2026-09-09

## Revision 3 checks — MIT licensing

Revision 3 adds the MIT License and updates licensing documentation and distribution hashes. It does not change executable code or tests.

- All 18 offline test cases passed again after extracting the MIT-licensed ZIP into a separate path containing spaces.
- All 50 protected research/evidence files and the recorded core source fingerprint still match.
- The final manifest covers 69 files; the ZIP contains those files plus the manifest itself (70 files total).
- The distribution checker confirms the license file is present. This presence check is not legal clearance.

## Revision 2 checks

Revision 2 fixes distribution-only tooling and notebook guidance; the research sources, configurations, and historical results are unchanged.

- The offline test suite now has **18 passing cases**, checked in the revision folder and again after extracting the final ZIP into a different path containing spaces.
- Checker regression cases cover a valid distribution, a missing manifest, and a changed file. Missing manifests and hash mismatches produce a failed report and a nonzero exit.
- Notebook regression cases cover both default settings and renamed output/experiment folders. Setup and the audit command use the same configuration-derived result path.
- The notebook's preservation guidance requires all four audit reports, regardless of the audit outcome.
- Distribution hashes, ZIP contents, and all 50 protected research/evidence files were checked again.

This revision did not repeat the initial wheel build/install or CLI planning checks listed below. Those checks were performed on the first distribution; the package sources and original command scripts remain byte-for-byte unchanged. The limitations in the "Not performed" section still apply.

## Initial distribution checks (before revision 2)

| Check | Observed result |
| --- | --- |
| Preserved research sources/evidence | 50 protected files matched their source SHA-256 values |
| Recorded core source-file set | 25 files; fingerprint matched the recorded experiment fingerprint |
| Offline pytest suite | 14 tests passed in the distribution checkout |
| Relocated checkout with spaces in its path | All 14 tests passed again |
| Command-line interfaces | Seven original command `--help` calls and the independent audit `--help` call succeeded |
| Full job planning from another working directory | 220 unique jobs printed; no training or download |
| Wheel build | `cei220` wheel built locally with no dependency downloads |
| Wheel installation | Installed in a newly created virtual environment using existing shared dependency packages |
| Installed-package smoke | Imported from the new environment, built 220 jobs, and computed a model-state hash |
| Notebook portability | Setup found the relocated root; fixed machine-specific paths and saved outputs were removed |
| Basic public-file checks | No common credential-pattern match; no model/checkpoint/private-key file included; no fixed home-directory path outside historical reference evidence |

## What the initial 14 tests mean

The original six unit tests cover repeatable seed derivation, specified namespace examples, model-state hashing behavior, run counts, and a test-only `block` event.

The eight added test cases cover canonical-versus-independent hashing for four tensor cases, CPU model-file loading, stable run-path reconstruction, relocated notebook setup, and a small offline pipeline. The offline pipeline executes seven one-epoch SmallCNN jobs using tiny synthetic CIFAR-shaped images and the original transforms. It checks aggregation, execution integrity, successful audit, deliberate model-file modification, and a deliberately missing model file. All outputs are temporary test fixtures, not research data.

The smoke configuration has only one run per strong-perturbation type. The integration test therefore expects execution integrity to pass and the full scientific expectations to remain unfulfilled; it does not relabel the smoke test as a successful 220-run scientific experiment.

## Local environment used for these checks

- Windows, Python 3.10.20
- PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128
- NumPy 2.2.5, pandas 2.3.3, PyYAML 6.0.2
- pytest 9.1.1, pip 25.1, setuptools 78.1.1, wheel 0.45.1

The synthetic training test selected the CPU. Installed dependency packages came from the existing local environment. The original research dependency files retain pytest 9.0.2; they were not changed to match this later validation environment.

## Not performed

- Installation of every dependency from the network into a completely fresh environment.
- Execution on a separate physical computer or Linux.
- Execution of the included manual GitHub Actions workflow.
- Real CIFAR-10/Fashion-MNIST downloads or a repeat of the historical 220-run experiment.
- Recalculation of hashes from the original 220 model files during distribution preparation.
- An exhaustive security audit or legal clearance for publication.

The existing 220-run audit reports remain historical evidence. Packaging checks and synthetic audit tests do not replace those reports or establish cross-environment reproducibility.

## Remaining publication decisions

The project code uses the MIT License. Repository creation is confirmed for https://github.com/LeeUiJeong/cei220. Final release metadata, dissertation citation details, and access to the full original model evidence are separate from this package check. Creation of the empty repository does not establish that these files have been uploaded.

## Repository upload copy — 2026-10-14

Revision 3 was independently checked again on this date: 18 offline tests passed; 50 protected files and the 25-file recorded core fingerprint matched; the original ZIP and its extracted 70-file folder matched exactly; and the command-line dry run printed 220 jobs from another working directory. Historical result/report tables were also cross-checked without loading the original model files.

Revision 4 changes repository/citation documentation only. The historical citation file is preserved at a new path. A regenerated manifest and the distribution checker verify this upload copy. Neither a repeated 220-run GPU experiment nor an original-model rehash is claimed.
