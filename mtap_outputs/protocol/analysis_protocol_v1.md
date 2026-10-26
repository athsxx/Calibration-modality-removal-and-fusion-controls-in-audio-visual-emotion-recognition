# MTAP revision analysis protocol, version 1

Locked: 29 September 2026, before the principal IEMOCAP logit export completed and before any RAVDESS or CREMA-D data were downloaded. Git branch `mtap-revision-20260929`, base commit `db0ad349`. Any later change is recorded as version 2 with its date and reason; version 1 is not edited.

## 1. Status of each evidence block

| Block | Status when this protocol was locked |
|---|---|
| IEMOCAP principal grid (5 systems x seeds {0,1,7} x 5 folds) | Clean macro-F1 already seen and published in the ESWA draft. All IEMOCAP inference below is **post hoc**. |
| IEMOCAP utterance-level logits, calibration, condition exports | Not yet seen for the multi-seed grid. |
| Selective-fusion method (Section 5) | Not implemented; no result seen. |
| RAVDESS-672 and CREMA-D four-class | Not downloaded; no result seen. Hypotheses in Section 6 are **prospective**. |

## 2. Units of analysis

- Primary unit for inference: **speaker**. IEMOCAP has 10 speakers (2 per session), RAVDESS 24 actors, CREMA-D 91 actors. Utterances are never treated as independent samples.
- Each utterance is predicted once per seed, by the model whose outer fold holds it out. Out-of-fold predictions are pooled across the five folds before computing a corpus-level metric.
- Seeds are a nuisance source of variance. Metrics are computed per seed and averaged across seeds inside every resampling replicate; seeds are not added as extra samples.

## 3. Metrics

Macro-F1 (primary), UAR, accuracy, per-class F1 and recall, NLL, Brier (sum over classes), ECE with 15 equal-width and 15 equal-mass bins, MCE, classwise ECE, AURC (area under the risk-coverage curve), and for each test condition: VRR, signed confidence change, and correctness transitions. VRR is `max(p_altered) > max(p_clean)`; exact ties are not violations. Temperature is fitted on clean development logits only (bounds 0.05-20) and applied unchanged to every test condition.

## 4. Inference procedures

1. **Speaker-cluster bootstrap** (B = 10,000, seed 20260929). Resample speakers with replacement, stratified by corpus session or fold group so every replicate still contains each outer test fold. The same speaker draw is applied to every model and seed in a comparison, so differences stay paired. Report percentile 95% CIs. With 10 IEMOCAP speakers these intervals are approximate; this is stated beside every IEMOCAP interval.
2. **Hierarchical bootstrap** (speakers, then utterances within speaker) as a sensitivity analysis.
3. **Fold-level paired t-test** on the five seed-averaged session differences, kept for continuity with the ESWA draft.
4. **Utterance-level GEE**: logistic regression of correctness on model indicator, exchangeable working correlation clustered by speaker, robust standard errors. Reported as an odds ratio with 95% CI.
5. **Equivalence (TOST)** for every claim of "no difference". Margin **delta = 0.02 macro-F1**, fixed here. Equivalence is declared only if the 90% speaker-bootstrap CI lies inside (-0.02, +0.02). A non-significant difference without equivalence is reported as "inconclusive", not "no difference".
6. **Variance decomposition** of run-level macro-F1 into fold, seed and residual components (random-effects ANOVA), to state what each reported SD summarises.
7. **Minimum detectable effect** for the n = 5 paired design at alpha 0.05 and power 0.80, using the observed SD of fold differences.
8. **Robustness slope**: macro-F1 versus visual token-loss fraction {0, 0.25, 0.5, 0.75}, least-squares slope per run, summarised with the speaker bootstrap.
9. **Cross-corpus synthesis**: per-corpus effect with CI, plus a DerSimonian-Laird random-effects estimate and I^2 across the three corpora. With three corpora the pooled estimate is descriptive; the per-corpus results carry the claim.

## 5. Selective-fusion method: evaluation rules fixed before implementation

- Anchor modality chosen per corpus and per outer fold from **development** macro-F1 only.
- All gate parameters are fitted on development predictions only. Test labels are never used for fitting, thresholds or model selection.
- Required comparators on identical folds and seeds: anchor-only, always-on fusion (ConcatMatched and Essence, both visual backbones), logit averaging, QMF-style energy-weighted late fusion, and CML-regularised ConcatMatched.
- Success criterion S1: clean macro-F1 higher than always-on AU fusion with speaker-bootstrap 95% CI excluding 0, **or** S2: equivalence to always-on AU fusion (Section 4.5) together with a VRR reduction under zero-filled video whose 95% CI excludes 0.
- If neither S1 nor S2 holds on IEMOCAP, the method is reported as a negative result and is not presented as the paper's contribution.

## 6. Prospective hypotheses for RAVDESS-672 and CREMA-D (Holm within each corpus)

H1. Always-on fusion with ImageNet ResNet-18 face features does not exceed the stronger unimodal model (TOST, margin 0.02, or a difference whose CI includes 0).
H2. Fusion with facial Action Unit features exceeds ResNet-18 fusion (ConcatMatched-AU vs ConcatMatched-ResNet).
H3. Essence-AU and ConcatMatched-AU are equivalent (TOST, margin 0.02).
H4. The stronger unimodal model differs across corpora (audio on IEMOCAP; reported, not assumed, on the other two).
H5. The selective-fusion method meets S1 or S2 (Section 5).
H6. Validation-only temperature scaling lowers ECE but does not lower VRR under zero-filled video.

Family: H1-H6, Holm within each corpus. Null and reversed results are reported in the main text.

## 7. Reporting rules

- Every table states its unit of resampling, number of speakers, and whether its family is post hoc or prospective.
- The 73 IEMOCAP runs with unknown non-finite history are included, flagged, and repeated in a sensitivity analysis restricted to runs with complete metadata where coverage allows.
- No number enters the manuscript unless it is regenerated from the identified exports by committed code.
