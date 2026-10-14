# Publication checklist

## Author decisions

- [x] Select the MIT License and include its full text as `LICENSE`.
- [ ] Confirm rights to publish and license the included material, including any institutional, collaborator, or third-party rights.
- [x] Confirm the author/copyright-holder spelling as `Uijeong Lee`.
- [x] Identify the source repository as https://github.com/LeeUiJeong/cei220 and add its URL to `CITATION.cff`.
- [x] Preserve historical citation metadata at `docs/ORIGINAL_CITATION.cff`.
- [ ] Add final dissertation citation details when available.
- [ ] Create and verify a GitHub release tag and release date; neither is asserted by this upload copy.
- [ ] Decide how readers can access full original model-state evidence. If unavailable, clearly state that this repository alone cannot re-audit the original model files.
- [ ] Review historical local-path and environment metadata under `reference_results/` before publication.

## Technical checks before claiming broader support

- [ ] Read `docs/VALIDATION.md` for performed and unperformed checks.
- [ ] Run the manual GitHub Actions workflow after repository publication is authorized. No remote CI run has occurred during preparation.
- [ ] Test dependency downloads and installation in a fully fresh environment before claiming this has been verified.
- [ ] Confirm final files with `python tools/check_distribution.py`.

Publish the folder structure, not only a ZIP. Keep datasets, fresh results, model/checkpoint files, credentials, caches, and local test environments out of the source repository. A ZIP may be a separately versioned download.

Preserve the original share archive and complete evidence directory independently. This distribution is not their backup, and deployment-only changes must not be described as the exact historical experiment version.
