# Amendment to analysis protocol (version 5): faithful CML

Dated 8 October 2026. Written after the fidelity audit (`mtap_outputs/cml_fidelity_audit.md`) and before any faithful-CML training, export, or test score. Versions 1–4 are unchanged. The 45 CML-inspired penalty runs stay as they are. This amendment does not retune them and does not select λ from their scores.

## 1. Why this run exists

The 5 October 2026 review (pages 3 and 5) says the uploaded CML training is an opt-in hinge, while Ma et al. (ICML 2023) Algorithm 1 also puts classification loss on full and sampled partial inputs. The audit confirms that difference. The existing grid is named a CML-inspired penalty. This grid is the faithful comparator, so claims about the published method are not carried by the penalty alone.

## 2. What is trained

Model: ConcatMatched-AU. Corpora: IEMOCAP (`artifacts`), RAVDESS (`artifacts_ravdess`), CREMA-D (`artifacts_cremad`). Seeds 0, 1, and 7. Folds 1–5. Tags `{corpus}_concat_matched_au_cmlfaithful_seed{S}`. Configs: `configs/{iemocap,ravdess,cremad}_w2v_video_au_cmlfaithful.yaml`. Each file is the corpus CML-inspired yaml plus `training.cml_variant: faithful`. Optimiser, learning rate 0.0003, patience 8, epoch budget 50, and development macro-F1 selection are unchanged.

`training.cml_weight` is 1.0. That is one fixed untuned setting. Appendix H.5 of Ma et al. selected λ on a validation set from {1, 5, 10, …, 100}. This study does not. The value is not an authors’ default.

Objective, from Algorithm 1 with M = 2: each update erases one modality, chosen uniformly, and minimises (1/2) (CE(full) + CE(partial)) + λ mean(relu(Conf(partial) − Conf(full))). Conf is the max softmax probability. Neither confidence is detached. Erasure is zero-fill of that modality’s features with the mask unchanged, the same operation as export `zero_audio` / `zero_video`. Branch-disabled evaluation is not the training removal. On IEMOCAP the valence–activation concordance (weight 0.2) remains on the full input only, outside the 1/2, as it does for ConcatMatched-AU and for the CML-inspired penalty. RAVDESS and CREMA-D have no such term.

The penalty code path is the default (`cml_variant: penalty`) and is not retrained.

## 3. Pairs, locked before test scores

Effects are model B minus model A. Pair files are not to be edited after results:

- `pairs_cmlfaithful_iemocap.json` (baseline name `concatmatched_au`)
- `pairs_cmlfaithful_ravdess.json` and `pairs_cmlfaithful_cremad.json` (baseline name `concat_matched_au`)

Families:

1. `cmlfaithful_vs_concat_au`: ConcatMatched-AU versus faithful CML.
2. `cmlfaithful_vs_cml_penalty`: CML-inspired penalty versus faithful CML.

## 4. Metrics and verdicts

Reported outcomes, each pair, each corpus:

- clean macro-F1
- zero-audio VRR and zero-video VRR
- branch-disabled-audio VRR and branch-disabled-video VRR
- equal-width ECE, raw and after temperature scaling
- NLL and Brier, raw and after temperature scaling

Temperature is fit on clean development data only, inside the existing strong analysis, separately for each model. VRR uses the analysis rule already in the code: a violation is a strict increase in max probability. Exact ties are not violations.

Verdicts are protocol v2 consensus, unchanged. **Different** only if the 95% speaker-bootstrap interval excludes 0 and the Holm-adjusted session-level paired t p is below 0.05. **Equivalent** only if both 90% intervals (speaker bootstrap and session-level t) lie inside (−0.02, +0.02), and only for macro-F1, UAR, and accuracy. **Inconclusive** otherwise. Equivalence is reported before difference when both apply. Holm is grouped by family, metric, condition, and calibration, so the two families are not pooled. ECE, NLL, Brier, and VRR are not equivalence-margin metrics. The resampling unit is the speaker (unstratified). Five sessions limit the paired t. Seeds are averaged within each held-out session before that test.

Null and reversed results are reported. These comparisons do not establish a clinical claim, and they do not replace the CML-inspired penalty results.
