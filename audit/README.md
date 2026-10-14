# Independent model-state rehash audit

The audit script is preserved byte-for-byte from the historical source. It loads saved model states on the CPU, recomputes their canonical content hashes, and compares them with `metrics.json`, `hashes.json`, and `aggregate_metrics.csv`. It also checks the stored model file's SHA-256. It does not retrain models or use recorded hashes as calculation inputs.

## Required inputs

Prepare the Python environment described in the repository README and a **working copy** of the complete run-result tree. Each run must contain `model_state_only.pt`, `metrics.json`, and `hashes.json`; the result root must contain `aggregate_metrics.csv`.

The included `reference_results/` summaries alone are insufficient. Full model files are not included, and no download URL has been assigned.

## Command line

From the project root, with the complete working results in the configured location:

```text
python audit/audit_model_state_hashes.py --project-root . --expected-runs 220
```

For complete working results stored elsewhere:

```text
python audit/audit_model_state_hashes.py --project-root . --result-root /path/to/complete-result-copy --expected-runs 220
```

Replace the example path with your actual result directory; quote paths containing spaces. Keep the dataset/model/group/mode/seed/run hierarchy.

## Notebook

Start JupyterLab inside the project and open `audit/CEI_220_Model_State_Rehash_Audit.ipynb`. Its setup cell finds the project root from the current working directory and reads `project.output_root` and `project.experiment_name` from `configs/experiment_220.yaml`. Setup, the audit command, and report reading use the same result path. It stops when the working result aggregate is missing.

## Outputs and preservation

The selected result directory receives these four reports:

- `model_state_rehash_audit.csv`
- `model_state_rehash_group_summary.csv`
- `model_state_rehash_mismatches.csv`
- `model_state_rehash_summary.json`

Keep and share all four reports regardless of whether the audit passes, including the mismatch report when no mismatches are found.

**Re-running overwrites these four reports. Use a working copy to retain historical evidence.** Missing files, loading errors, or mismatches produce a nonzero exit after reports are written. Passing requires the expected run count and all recorded comparisons to agree.

The original script includes a loading fallback for older PyTorch versions. Only use trusted model files. Hash agreement establishes consistency with supplied records, not correctness of every training operation or predictive performance.
