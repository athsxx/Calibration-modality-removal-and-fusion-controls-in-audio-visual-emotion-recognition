# Faithful CML results

Dated 8 October 2026. The 45-run faithful grid finished at 23:10 with zero failures. Speaker-level analyses are `mtap_outputs/analysis_cmlf_{iemocap,ravdess,cremad}/`. Effects are model B minus model A. Model B is `concat_matched_au_cmlfaithful`. Model A is `concatmatched_au` on IEMOCAP and `concat_matched_au` on RAVDESS and CREMA-D for family `cmlfaithful_vs_concat_au`, and `concat_matched_au_cml` for family `cmlfaithful_vs_cml_penalty`. Displays below are four decimal places, half-up, from `effect_b_minus_a` or from `estimate`. Verdicts are the stored v2 `consensus_verdict`. No number below was recomputed from predictions.

## Fidelity difference

The audit is `mtap_outputs/cml_fidelity_audit.md`. Ma et al. (ICML 2023) Algorithm 1, for two modalities, minimises

L = (1/2) (CE(full) + CE(one partial)) + λ mean(relu(Conf(partial) − Conf(full))),

with one modality erased at random on each update and with no stop-gradient. The 45-run grid already in the manuscript is not that objective. It is a CML-inspired penalty: task loss on the full input only, plus the sum of two batch-mean hinges after zero-filling audio and video. λ = 1.0 on both variants is one fixed untuned setting. It is not an authors’ default. Appendix H.5 of Ma et al. selected λ on validation data. On IEMOCAP the valence–activation term (weight 0.2) stays on the full input only, outside the 1/2, for the baseline, the penalty, and the faithful run. RAVDESS and CREMA-D have no valence–activation term. Removal in both variants is zero-fill of features with the mask unchanged. That is export `zero_audio` / `zero_video`. It is not `branch_disabled_audio` or `branch_disabled_video`.

## Point estimates, clean

Speaker-pooled clean estimates from `point_and_ci_metrics.csv`. Macro-F1 does not change with temperature scaling.

| Corpus | System | Macro-F1 | ECE raw | ECE temp. | NLL raw | NLL temp. | Brier raw | Brier temp. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| IEMOCAP | ConcatMatched-AU | 0.6107 | 0.1433 | 0.0315 | 1.1056 | 0.9474 | 0.5585 | 0.5223 |
| IEMOCAP | CML-inspired | 0.6042 | 0.1328 | 0.0341 | 1.0841 | 0.9550 | 0.5602 | 0.5280 |
| IEMOCAP | Faithful | 0.6168 | 0.1273 | 0.0282 | 1.0849 | 0.9397 | 0.5499 | 0.5180 |
| RAVDESS | ConcatMatched-AU | 0.7666 | 0.1676 | 0.0475 | 1.2266 | 0.5885 | 0.3679 | 0.3073 |
| RAVDESS | CML-inspired | 0.7783 | 0.1397 | 0.0591 | 0.9858 | 0.5948 | 0.3320 | 0.2980 |
| RAVDESS | Faithful | 0.7726 | 0.1428 | 0.0623 | 1.0404 | 0.6062 | 0.3430 | 0.3046 |
| CREMA-D | ConcatMatched-AU | 0.8016 | 0.1283 | 0.0189 | 0.8613 | 0.5177 | 0.3189 | 0.2768 |
| CREMA-D | CML-inspired | 0.8039 | 0.1312 | 0.0268 | 1.0607 | 0.5417 | 0.3278 | 0.2858 |
| CREMA-D | Faithful | 0.8117 | 0.1016 | 0.0168 | 0.6744 | 0.5017 | 0.2927 | 0.2675 |

`point_and_ci_metrics.csv` does not contain VRR. VRR below is the paired effect only.

## Faithful minus ConcatMatched-AU

| Outcome | IEMOCAP | RAVDESS | CREMA-D |
|---|---|---|---|
| Clean macro-F1 | +0.0060 inconclusive | +0.0060 inconclusive | +0.0101 inconclusive |
| Zero-audio VRR, raw | −0.3129 different | −0.0541 inconclusive | −0.1262 different |
| Zero-audio VRR, temp. | −0.3265 different | −0.0625 inconclusive | −0.1267 different |
| Zero-video VRR, raw | −0.0950 different | −0.2326 different | −0.1410 different |
| Zero-video VRR, temp. | −0.1134 different | −0.2391 different | −0.1484 different |
| Branch-audio VRR, raw | −0.0145 inconclusive | −0.0060 inconclusive | −0.0131 inconclusive |
| Branch-audio VRR, temp. | −0.0081 inconclusive | −0.0005 inconclusive | −0.0089 inconclusive |
| Branch-video VRR, raw | +0.0996 different | −0.0486 inconclusive | −0.0263 inconclusive |
| Branch-video VRR, temp. | +0.1140 different | −0.0298 inconclusive | −0.0194 inconclusive |
| ECE raw | −0.0160 inconclusive | −0.0248 inconclusive | −0.0266 inconclusive |
| ECE temp. | −0.0034 inconclusive | +0.0148 inconclusive | −0.0021 inconclusive |
| NLL raw | −0.0208 inconclusive | −0.1862 inconclusive | −0.1870 inconclusive |
| NLL temp. | −0.0076 inconclusive | +0.0177 inconclusive | −0.0159 inconclusive |
| Brier raw | −0.0086 inconclusive | −0.0249 inconclusive | −0.0261 inconclusive |
| Brier temp. | −0.0043 inconclusive | −0.0027 inconclusive | −0.0093 inconclusive |

A negative VRR, ECE, NLL, or Brier effect means the faithful run is lower. Lower VRR, ECE, NLL, and Brier are the favourable directions. Clean macro-F1 is inconclusive on all three corpora. Several intervals exclude 0 and the Holm session-level test does not, including CREMA-D clean macro-F1 and RAVDESS and CREMA-D raw ECE. Those rows stay inconclusive.

## Faithful minus CML-inspired

| Outcome | IEMOCAP | RAVDESS | CREMA-D |
|---|---|---|---|
| Clean macro-F1 | +0.0126 different | −0.0057 inconclusive | +0.0078 equivalent |
| Zero-audio VRR, raw | +0.0890 different | +0.2560 different | +0.1138 different |
| Zero-audio VRR, temp. | +0.0666 different | +0.2505 different | +0.0865 different |
| Zero-video VRR, raw | +0.1566 different | −0.0040 inconclusive | +0.1061 different |
| Zero-video VRR, temp. | +0.1470 different | +0.0208 inconclusive | +0.0910 different |
| Branch-audio VRR, raw | +0.0136 inconclusive | +0.0680 inconclusive | −0.0049 inconclusive |
| Branch-audio VRR, temp. | +0.0140 inconclusive | +0.0823 inconclusive | −0.0016 inconclusive |
| Branch-video VRR, raw | +0.0602 inconclusive | −0.0045 inconclusive | −0.0227 inconclusive |
| Branch-video VRR, temp. | +0.0659 inconclusive | +0.0099 inconclusive | −0.0194 inconclusive |
| ECE raw | −0.0055 inconclusive | +0.0032 inconclusive | −0.0296 different |
| ECE temp. | −0.0059 inconclusive | +0.0032 inconclusive | −0.0099 inconclusive |
| NLL raw | +0.0007 inconclusive | +0.0546 inconclusive | −0.3863 different |
| NLL temp. | −0.0152 inconclusive | +0.0114 inconclusive | −0.0400 different |
| Brier raw | −0.0104 inconclusive | +0.0110 inconclusive | −0.0350 different |
| Brier temp. | −0.0100 different | +0.0066 inconclusive | −0.0183 inconclusive |

On zero-fill VRR, a positive effect means the faithful run violates ranking more often than the inspired penalty. That is the consensus result for both modalities on IEMOCAP and CREMA-D, and for zero-audio on RAVDESS. RAVDESS zero-video is inconclusive. Branch-disabled VRR does not copy that pattern: every branch-disabled contrast against the penalty is inconclusive, and against ConcatMatched-AU the only consensus difference is IEMOCAP branch-disabled video, where the faithful VRR is higher.

Temperature-scaled ECE versus ConcatMatched-AU is inconclusive on all three corpora. Versus the penalty it is also inconclusive, including CREMA-D (−0.0099), where the raw ECE contrast is different and the faithful run is lower. CREMA-D raw NLL, raw Brier, and temperature-scaled NLL are different in the faithful direction (lower). IEMOCAP temperature-scaled Brier is different and lower. Those rows do not make a general calibration claim. CREMA-D temperature-scaled Brier remains inconclusive.

## What the paper may claim

- The reported 45-run grid is a CML-inspired penalty, not Algorithm 1. The faithful run is the Algorithm 1 comparator at the same fixed untuned λ = 1.0.
- At that weight, the inspired penalty reduced zero-fill VRR on both modalities on all three corpora. Raw ECE fell numerically on IEMOCAP and RAVDESS. After development-only temperature scaling, ECE differences were inconclusive on those two corpora and worse on CREMA-D. The calibration conclusion depends on the metric. That wording is the 5 October review, and it still describes the inspired penalty.
- The faithful comparator lowered zero-fill VRR relative to ConcatMatched-AU on IEMOCAP and CREMA-D for both modalities, and on RAVDESS for zero-video only. RAVDESS zero-audio VRR is inconclusive. Temperature-scaled ECE versus ConcatMatched-AU is inconclusive on all three corpora. Clean macro-F1 versus ConcatMatched-AU is inconclusive on all three corpora.
- Where faithful and inspired zero-fill VRR differ, the inspired penalty is lower. The faithful run is not a second copy of the penalty’s VRR result.
- Branch-disabled removal is a different operation and does not show the zero-fill VRR pattern. IEMOCAP branch-disabled video VRR is higher under the faithful run than under ConcatMatched-AU.
- The paper may not call the ranking-versus-calibration distinction its novelty. Ma et al. already made it. The contribution is the controlled three-corpus comparison, including where the two objectives and the two removal operations disagree.
- The paper may not say that λ = 1.0 is the authors’ default, that the whole study was pre-registered, that parameter count explains the Essence comparison, that equivalence inside ±0.02 is universal, that an inconclusive RAVDESS contrast is equivalence, or that all 435 exported runs have verified training histories.
