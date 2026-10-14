# Historical reference summaries

`cei_220_reference_v1/` contains 18 CSV/JSON files from the existing experiment and independent audit. Their bytes were preserved; these are historical reference records, not newly executed results.

- `aggregate_metrics.csv`: individual run metrics.
- `summary_by_condition.csv`: condition summaries that may pool external seeds.
- `seed_generalization_by_seed.csv`: within-external-seed repeat summaries.
- `verification_report.json`: separate execution-integrity and scientific-expectation results.
- `model_state_rehash_summary.json`: historical independent audit summary.
- `model_state_rehash_audit.csv`, `model_state_rehash_group_summary.csv`, `model_state_rehash_mismatches.csv`: audit details and exceptions.

Other files retain environment, artifact-hash, event, table, and run-manifest summaries. Original absolute paths are retained as provenance, not portable execution instructions. Stable dataset/model/group/mode/seed/run identifiers locate corresponding original run files.

Do not write training or analysis output here. Fresh results belong under `results/`. Model files, checkpoints, full logs, and datasets are not included. Reading these reports is not a new independent model-state rehash audit.

Access to the full original evidence has not been configured. The author must supply an actual access method before advertising those model files as available to readers.
