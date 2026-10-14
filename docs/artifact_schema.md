# Artifact Schema

Each completed run creates:

- `metadata.json`
- `seed_transparency_log.jsonl` (all non-baseline modes)
- `policy_log.jsonl`
- `metrics.json`
- `model_state_only.pt`
- `checkpoint.pt`
- `hashes.json`
- `run_status.json`
- `stdout.log`
- `stderr.log`

Aggregate outputs include:

- `aggregate_metrics.csv`
- `artifact_hashes.csv`
- `summary_by_condition.csv`
- `seed_generalization_by_seed.csv`
- `hash_summary_by_artifact.csv`
- `environment_by_run.csv`
- `environment_summary.csv`
- `policy_events.csv`
- `table_main_kr.csv`
- `table_seed_generalization_kr.csv`
- `table_strong_perturbation_kr.csv`
- `analysis_summary.json`
- `verification_report.json`
