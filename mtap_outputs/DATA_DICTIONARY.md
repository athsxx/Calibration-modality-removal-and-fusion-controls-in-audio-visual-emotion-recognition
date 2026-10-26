# Data dictionary

Column names below are the headers in the files on disk. Effects are model B minus model A. Blank CI cells mean that interval was not written. `history_unknown` is not zero. Probabilities and rates are fractions. Macro-F1, UAR, accuracy, VRR, and ECE are on that fraction scale. NLL and sum-of-classes Brier are the companion's scores, not rescaled. Seed-level SDs in `analysis/summary_scores.csv` use ddof = 0.

## Labels and names

IEMOCAP class order, from `analysis_strong_v2/per_class.csv`: 0 angry, 1 happy_excited, 2 neutral, 3 sad.

RAVDESS and CREMA-D use the same integers. Class 1 is happy on those corpora, not IEMOCAP's combined happy/excited. That mapping is a within-corpus label, not a claim of cross-dataset transfer.

IEMOCAP model ids: `audio_w2v`, `concatmatched_v2`, `essence_v2`, `concatmatched_au`, `essence_au`.

New-corpus model ids: `audio`, `video_au`, `video_resnet`, `concat_matched_au`, `concat_matched_resnet`, `essence_au`, `essence_resnet`.

Selective-fusion ids in `selective_v3_*/selective_export/point_summary.csv`: `selective_cmau`, `selective_esau`, `logit_avg_av`, `qmf_av`, plus the anchor and always-on systems.

Conditions that were exported: `clean`, `zero_audio`, `zero_video`, `branch_disabled_audio`, `branch_disabled_video`, `video_token_loss_25`, `video_token_loss_50`, `video_token_loss_75`. Audio-only runs do not have video conditions. Video-only runs do not have audio-branch conditions. `fully_masked_audio` and `fully_masked_video` were not implemented.

Calibration: `raw` is the softmax of the stored logits. `temperature` applies one positive scalar fit on clean development logits only (bounds 0.05–20). Test labels were not used to fit it.

Consensus, in `analysis_strong_v2` and the new-corpus paired files: `different` only if the 95% speaker-bootstrap CI excludes 0 and the Holm session-level paired t p < 0.05; `equivalent` only if both 90% intervals lie inside ±0.02; otherwise `inconclusive`.

## Prediction exports

`exports/**/validation_predictions.csv.gz` and `test_predictions.csv.gz` (same columns in the RAVDESS, CREMA-D, and video-only AU exports):

`dataset`, `model`, `seed`, `fold`, `run_id`, `utterance_id`, `condition`, `split`, `label`, `p_0`, `p_1`, `p_2`, `p_3`, `z_0`, `z_1`, `z_2`, `z_3`, `video_valid_tokens`, `video_kept_tokens`, `corruption_seed`

`p_*` are probabilities. `z_*` are original logits. `split` is `validation` or `test`. Validation files are clean only.

`audit/run_manifest.csv` and `exports/run_manifest.csv`:

`model`, `model_name`, `seed`, `fold`, `run_id`, `checkpoint_relative_path`, `config_relative_path`, `audio_dir`, `video_dir`, `feature_manifest_sha256`, `training_commit`, `historical_feature_sanitized_count`, `config_basis`, `status`, `rejected_reasons`, `unknown_history`, `checkpoint_sha256`, `metrics_sha256`, `config_sha256`, `audio_max_tokens`, `video_max_tokens`, `checkpoint_stage`, `best_epoch`, `stage2_reached`, `used_stage1_fallback`, `nonfinite_batch_count`, `reference_macro_f1`, `reference_uar`, `audio_dim`, `video_dim`, `reference_predictions_relative_path`, `reference_predictions_sha256`

`training_commit` for the historical IEMOCAP checkpoints is the string `unknown (not stored in checkpoint)`.

`audit/feature_manifest.csv`: `cache`, `utterance_id`, `relative_path`, `sha256`, `tokens`, `dimension`, `status`

`audit/split_manifest.csv`: `dataset`, `fold`, `split`, `utterance_id`, `session`, `speaker_id`, `label`

Each run's `complete.json` includes `parity_absolute_difference`, `history_status`, `conditions`, and `clean_parity_basis`.

## Companion analysis (`analysis/`)

`fold_scores.csv`: `dataset`, `model`, `seed`, `fold`, `history_status`, `condition`, `calibration`, `n`, `macro_f1`, `uar`, `accuracy`, `nll`, `brier`, `ece_width`, `ece_mass`, `mce_width`, `mean_confidence`, `confidence_minus_accuracy`

`summary_scores.csv`: `model`, `condition`, `calibration`, `n_runs`, `complete_15_run_coverage`, then mean and ddof-0 SD across the three seed means for macro-F1, UAR, accuracy, NLL, Brier, ECE width, ECE mass, and MCE width. Column names end in `_mean_of_seed_means` or `_sd_seed_means_ddof0`.

`paired_conditions.csv`: `dataset`, `model`, `seed`, `fold`, `history_status`, `condition`, `calibration`, `n`, `vrr`, `mean_clean_minus_altered_confidence`, `clean_correct_altered_wrong_fraction`, `clean_wrong_altered_correct_fraction`, `prediction_change_fraction`, `macro_f1_drop`, `uar_drop`

VRR is the fraction of paired utterances with `max(p_altered) > max(p_clean)`. Exact ties are not violations.

`temperatures.csv`: `dataset`, `model`, `seed`, `fold`, `history_status`, `temperature`, `n_validation`, `validation_nll_before`, `validation_nll_after`, `boundary`

`comparison_tests.csv`: `comparison`, `model_a`, `model_b`, `effect_b_minus_a`, `n_session_folds`, `raw_p`, `ci95_t_low`, `ci95_t_high`, `family`, `note`, `holm_p`

Family value: `historical_seven_posthoc`. Five session differences, not 15 independent seeds.

`paired_fold_differences.csv`: `comparison`, `model_a`, `model_b`, `fold`, `difference_b_minus_a`

## Speaker-level analysis

Same headers in `analysis_strong_v2/`, `analysis_strong_ravdess/`, and `analysis_strong_cremad/`.

`point_and_ci_metrics.csv`: `model`, `condition`, `calibration`, `metric`, `estimate`, `ci95_low`, `ci95_high`, `ci90_low`, `ci90_high`, `ci_source`, `n_utterances`, `n_speakers`, `n_seeds`, `n_folds`, `partial_folds`

Metrics include `macro_f1`, `uar`, `accuracy`, `nll`, `brier`, `ece_width`, `ece_mass`, `mce_width`, `classwise_ece`, `aurc`. `ci_source` is `speaker_bootstrap`.

`paired_comparisons.csv`: `model_a`, `model_b`, `family`, `condition`, `calibration`, `metric`, `effect_b_minus_a`, `ci95_low`, `ci95_high`, `ci90_low`, `ci90_high`, `bootstrap_p`, `holm_bootstrap_p`, `bootstrap_verdict`, `fold_t_verdict`, `consensus_verdict`, `fold_t_effect`, `fold_t_df`, `fold_t_p`, `holm_t_p`, `fold_t_ci95_low`, `fold_t_ci95_high`, `fold_t_ci90_low`, `fold_t_ci90_high`, `fold_t_note`, `gee_or`, `gee_ci95_low`, `gee_ci95_high`, `gee_p`, `gee_covariance`, `gee_error`, `gee_robust_or`, `gee_robust_p`

IEMOCAP family: `iemocap_historical_posthoc`. New corpora family: `prospective_v3`.

`analysis_strong/paired_comparisons.csv` is the older file. It has `tost_verdict` and does not have `consensus_verdict`. Do not quote it as the v2 consensus.

`condition_effects.csv`: `model`, `condition`, `calibration`, `metric`, `estimate`, `ci95_low`, `ci95_high`, `ci90_low`, `ci90_high`, `partial_folds`

`per_class.csv`: `model`, `condition`, `calibration`, `class_index`, `class_name`, `f1`, `f1_ci95_low`, `f1_ci95_high`, `recall`, `recall_ci95_low`, `recall_ci95_high`, `partial_folds`

`hierarchical_sensitivity.csv`: `analysis`, `model`, `model_a`, `model_b`, `family`, `condition`, `calibration`, `metric`, `estimate`, `ci95_low`, `ci95_high`

`stratified_sensitivity.csv`: `model_a`, `model_b`, `family`, `condition`, `calibration`, `metric`, `resampling`, `effect_b_minus_a`, `ci95_low`, `ci95_high`, `ci90_low`, `ci90_high`, `bootstrap_p`

`robustness_slopes.csv`: `kind`, `model`, `model_a`, `model_b`, `family`, `slope`, `ci95_low`, `ci95_high`, `ci90_low`, `ci90_high`

`variance_components.csv`: `model`, `n_folds`, `n_seeds`, `ss_fold`, `ss_seed`, `ss_residual`, `var_fold`, `var_seed`, `var_residual`, `negative_truncated`, `note`

`minimum_detectable_effects.csv`: `model_a`, `model_b`, `metric`, `condition`, `calibration`, `n_folds`, `sd_fold_diff`, `mde`, `note`

`selective_v3_*/selective_export/point_summary.csv`: `model`, `condition`, `macro_f1`, `vrr`

`vrr` is blank on `clean` rows.

`figures/figure_source_data.csv`: `figure`, `series`, `model`, `condition`, `calibration`, `metric`, `estimate`, `ci95_low`, `ci95_high`. Copied from the analysis CSVs by `scripts/mtap_make_figures.py`.
