# Distribution preparation — 2026-09-09

This is a local distribution candidate, not a published release. Original package and citation versions have not been renumbered.

## Preserved

- Original `src/`, `scripts/`, `configs/`, `pyproject.toml`, `requirements-lock.txt`, `environment.yml`, four unit-test files, and `CITATION.cff`.
- Original `audit/audit_model_state_hashes.py`.
- All 18 existing CSV/JSON summary, verification, and audit files, relocated without content changes to `reference_results/cei_220_reference_v1/`.

Recorded code fingerprint: `621ff61f6bc58102025f5cb5abdc1cff9ea49e3e275145e288a7cbe6b6819a10`.
It covers selected files in `src/`, `scripts/`, `configs/`, `pyproject.toml`, and `requirements-lock.txt`, not the whole repository. Installation-generated files can affect a new fingerprint; the distribution checker also checks the exact recorded source-file set.

## Initial distribution-only changes (before license selection)

- Portable English/Korean instructions, pending-license notice, publication checklist, and validation documentation.
- Project-root discovery in the audit notebook, with saved outputs/execution counts cleared. The original notebook remains in the separate source share archive.
- Historical reference summaries separated from fresh runtime outputs.
- `.gitignore`, byte-preserving `.gitattributes`, manual GitHub Actions test workflow, additional offline tests, and source-integrity checker.
- Older upload checklist revised to match this layout.

`docs/SOURCE_PROVENANCE.json` identifies the original share archive and its file hashes. Its old bundle manifest and sharing guide are not manifests for this modified copy. `DISTRIBUTION_MANIFEST.json` describes the final distribution instead.

No historical numerical value or model-state hash was edited. No license or public repository was created in that initial preparation. Synthetic test runs are not additions to the dissertation's 220-run evidence set.

## Revision 2 — 2026-09-09

- The distribution checker now fails if `DISTRIBUTION_MANIFEST.json` is missing and reports whether the manifest is present.
- The audit notebook resolves the output folder and experiment name from its YAML configuration during setup. The audit command and report-reading cells use that same result path.
- Notebook guidance now requires preserving all four audit reports for both successful and unsuccessful audits, with a reminder that reruns overwrite reports.
- Added regression coverage for missing manifests, changed files, and custom result paths; 18 offline test cases pass.
- Updated the provenance hashes and final distribution manifest. The prior ZIP remains a separate, unchanged archive.

These changes do not modify training, seed derivation, policy execution, canonical or independent hash algorithms, original test files, or historical numerical/audit results. License selection was still pending in revision 2.

## Revision 3 — MIT license selection — 2026-09-09

- Added the full MIT License for project code and accompanying software documentation at the author's request.
- Updated the English/Korean README, license-scope notice, and publication checklist; software licensing does not assign new terms to historical research data/reports or third-party materials.
- Citation requests remain separate from the MIT license conditions.
- Updated source-provenance and distribution hashes. Original package metadata, citation file, all research sources, and historical evidence remain byte-for-byte unchanged.
- Rechecked all 50 protected files, the final ZIP, and 18 offline test cases after fresh extraction.

The previous archives remain unchanged. GitHub publication, final citation/repository details, and rights review remain separate tasks.

## Revision 4 — repository information — 2026-10-14

- Prepared a separate upload copy for https://github.com/LeeUiJeong/cei220.
- Added the repository URL to the current citation metadata while retaining software version 1.0.0. No GitHub release date, tag, DOI, or final dissertation citation is invented.
- Preserved the original citation file byte-for-byte at docs/ORIGINAL_CITATION.cff and recorded its new preservation location in source provenance.
- Updated repository and citation explanations and rebuilt the distribution manifest.
- Research code, configurations, environment files, tests, independent audit code, and all 18 historical result/report files are unchanged from revision 3. The old revision 3 folder and ZIP remain unchanged.

The preparation date does not establish that the files were uploaded or that a versioned GitHub release was published. Repository contents and the final commit must be verified after upload.
