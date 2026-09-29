# Results

Outputs behind the tables of the paper. All FRD test metrics use the operating
threshold selected on the validation partition; validation metrics are
reported at the 0.5 threshold, which is the selection criterion. Predictions
list `sample_id`, the true label, and the predicted probability of the
machine-generated class.

| Paper item | File(s) |
|---|---|
| Table I: baseline settings | `../scripts/FRD_Run_NonTransformer_Study.py` |
| Table II: Transformer backbones and fine-tuning settings | `../scripts/FRD_Run_Transformer_Study.py` (embedded runtime: model registry and configuration) |
| Table III: validation and test results of all systems | `frd_benchmark/all_systems.csv` |
| Section V-A: equal versus weighted voting of the three selected triples | `frd_benchmark/all_systems.csv` (rows E1–E3) |
| Table IV: stage progression, confusion counts, paired tests | `statistics/stage_selection_and_metrics.csv`, `statistics/stage_confidence_intervals.csv`, `statistics/planned_progression_paired_tests.csv`, `statistics/planned_progression_bootstrap_effects.csv` |
| Table V: sentiment ablation, first run (seed 42) | `statistics/sentiment_ablation_paired_tests.csv`, `statistics/sentiment_ablation_bootstrap_effects.csv` |
| Table V: sentiment ablation, other seeds and five-run summaries | `seed_robustness/per_seed_pair_differences.csv`, `seed_robustness/pair_summary.csv`, `seed_robustness/per_seed_model_metrics.csv` |
| Section IV: protocol of the repeated runs | `seed_robustness/protocol.json`, `seed_robustness/stage2_decision.json`, `seed_robustness/protocol_amendment_1.json`, `seed_robustness/protocol_amendment_2.json` |
| Table VI: AiGen-FoodReview zero-shot, first run | `external_aigen/all_systems_primary_metrics_with_ci.csv` |
| Table VI, last column; Section V-C: AiGen ROC-AUC over five runs | `seed_robustness/aigen_per_seed.csv`, `seed_robustness/aigen_seed_summary.csv` |
| Section V-C: Spearman correlations, FRD accuracy vs. AiGen ROC-AUC | `seed_robustness/spearman_frd_vs_aigen.csv` (from `../scripts/FRD_Seed_Correlations.py`) |
| Section III-A: AiGen checksum and overlap with FRD | `external_aigen/dataset_audit.json` |
| Section IV: backbone representatives | `selection/representative_selection.csv`, `selection/representative_selection.json` |
| Section IV: ensemble search and final system | `selection/ensembles/triple_search_validation.csv`, `selection/ensembles/selection_manifest.json` |
| Section IV: ModernBERT CPU versus GPU | `checks/modernbert_device_check.json` |
| Section V-C: 0.5 versus transferred thresholds | `external_aigen/all_systems_default_05_metrics.csv`, `external_aigen/diagnostics/operating_point_check.csv` |
| Sections III-A, V-C: truncation, surface features, stratified ROC-AUC | `external_aigen/diagnostics/truncation.csv`, `external_aigen/diagnostics/stratified_auc.csv`, `external_aigen/diagnostics/summary.json` (format-restricted subset: all 1,986 authentic reviews and the 483 generated reviews without a title line or line break; columns `auc_format_free`, `ff_ci_low`, `ff_ci_high`) |
| Section VI: OpSpam zero-shot | `external_opspam/<model>/metrics.json` (key `frd_validation_calibrated`) |
| Selection records and numeric precision | `checks/records_check.json` |

## Folders

### `frd_benchmark/`
`all_systems.csv` gives, for every classifier and ensemble, validation
accuracy, F1, and ROC-AUC; the selected threshold; test accuracy, precision,
recall, F1, ROC-AUC, and MCC; and the full confusion counts (TN, FP, FN, TP;
positive class = machine-generated). `non_transformers/` and `transformers/`
hold each system's `metrics.json` and its validation and test predictions.
`transformers/deberta_v3_base/reconciliation.json` records how the
predictions of plain DeBERTa-v3 were regenerated from its saved weights.

### `selection/`
Validation-based choice between the plain and the sentiment-augmented variant
of each backbone; the ranking of all 35 three-member ensembles; and, for the
three highest-ranked triples, the equal and weighted soft votes with their
weights, thresholds, and validation and test predictions.

### `statistics/`
Paired McNemar tests with Holm correction and paired bootstrap intervals
(1,000 resamples) for the stage progression and the sentiment ablation;
`selection_audit.json` records that all selections used validation data only.

### `external_aigen/`
Metrics and predictions of every Transformer system and ensemble on the
AiGen-FoodReview test split, with bootstrap intervals and the diagnostics.
`models/` holds the first runs (seed 42); `seed_<N>/models/` holds the
repeated runs, scored under the same frozen protocol.

### `seed_robustness/`
The repeated training runs of the 14 Transformer configurations. The protocol
files record the seed list, the staged extension rule, and the two amendments
before the runs they cover. `frd/seed_<N>/<model>/` holds each repeated run's
`metrics.json` and its validation and test predictions (same layout as
`frd_benchmark/transformers/`). For plain DeBERTa at seeds 123 and 2025,
`bf16_attempt_error.json` records the bfloat16 overflow that led to
amendment 2. The CSV files give per-run metrics, the
sentiment-minus-plain differences with unadjusted McNemar tests, five-run
summaries, the AiGen ROC-AUC of every run, and the Spearman correlations.
Local paths in the records are shown as `<project>`.

### `external_opspam/`
Zero-shot metrics on OpSpam, reported for transparency. OpSpam's deceptive
class is written by people, so it does not test the FRD task.

### `checks/`
Selection records collected from the saved outputs and the CPU/GPU
consistency check for ModernBERT.

## Not included

Model weights, checkpoints, and training logs are not stored; they are
regenerated by the scripts.
