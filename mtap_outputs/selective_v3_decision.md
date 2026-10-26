# Selective fusion decision (protocol v3, WP5)

Dated 1 October 2026, 01:40 IST. Runs: `scripts/mtap_selective_pilot.py` with `--boot 10000 --hboot 2000`, unstratified speaker bootstrap. Late fusion was **computed** on every corpus (log line `Late fusion: computed`; no `late_fusion_baselines.txt`). Test labels were not used to fit the gate, temperatures, tau, or the logit-average weight. The gate, tau grid, and success criteria were not changed. RAVDESS and CREMA-D name ConcatMatched-AU `concat_matched_au`; the pilot maps that export name onto the protocol system and does not rename original files.

Outputs: `mtap_outputs/selective_v3_{iemocap,ravdess,cremad}/`. The pilot refuses to write under `mtap_outputs`, so each run was written on `<DATA_ROOT>/wp5_selective_staging/` and the finished tree was moved. `combined_export/export_manifest.json` still records the staging path in `combined_from` so its SHA-256 matches `analysis/strong_analysis_manifest.json`.

Anchor (same model on all five folds, seed-averaged development macro-F1): IEMOCAP `audio_w2v`, RAVDESS `video_au`, CREMA-D `audio`.

Clean macro-F1 is the seed-averaged, speaker-pooled argmax score in `selective_export/point_summary.csv`. Zero-video VRR uses each system's development temperature. Effects below are selective minus the named comparator.

## Point estimates

| Corpus | Anchor | Always-on CMAU | Selective CMAU | Logit average | QMF | Always-on Essence | Selective Essence |
|---|---:|---:|---:|---:|---:|---:|---:|
| IEMOCAP | 0.578 | 0.611 | 0.618 | 0.605 | 0.603 | 0.613 | 0.619 |
| RAVDESS | 0.702 | 0.767 | 0.750 | 0.773 | 0.774 | 0.789 | 0.783 |
| CREMA-D | 0.719 | 0.802 | 0.808 | 0.813 | 0.806 | 0.808 | 0.808 |

Zero-video VRR: IEMOCAP always-on CMAU 0.296 vs selective 0.234; Essence 0.354 vs 0.284. RAVDESS 0.328 vs 0.294; Essence 0.423 vs 0.352. CREMA-D 0.249 vs 0.219; Essence 0.333 vs 0.304.

## How S1 and S2 are read

S1, protocol v1 §5: selective clean macro-F1 higher than always-on AU fusion, speaker-bootstrap 95% CI excludes 0 (`s1_bootstrap`).

S2, protocol v1 §5 citing §4.5: 90% speaker-bootstrap CI for that clean difference lies inside (−0.02, +0.02), and zero-video VRR is lower with a 95% CI that excludes 0 (`s2_bootstrap`).

Version 2 consensus, reported separately: **different** only if the 95% bootstrap CI excludes 0 and the Holm session-level paired t p < 0.05; **equivalent** only if both 90% intervals lie inside ±0.02; otherwise **inconclusive**. Equivalence is reported before a small difference. `s2_consensus` requires consensus equivalence on clean macro-F1 and a consensus-different VRR reduction.

## Primary: selective ConcatMatched-AU vs always-on ConcatMatched-AU

| Corpus | Δ clean macro-F1 (95% CI) | S1 | S2 (§4.5) | Consensus, clean macro-F1 | Consensus, zero-video VRR | s2_consensus |
|---|---|---|---|---|---|---|
| IEMOCAP | +0.007 (0.002, 0.012) | yes | yes | equivalent | inconclusive (Δ VRR −0.061) | no |
| RAVDESS | −0.017 (−0.029, −0.005) | no | no | inconclusive | inconclusive (Δ VRR −0.034) | no |
| CREMA-D | +0.007 (0.000, 0.013) | yes | yes | equivalent | inconclusive (Δ VRR −0.031) | no |

On IEMOCAP and CREMA-D the bootstrap flags for S1 and S2 are true, and the v2 rule reports the clean difference as **equivalent**, not as a gain. On RAVDESS selective fusion is lower than always-on ConcatMatched-AU. That is a negative result. The zero-video VRR drop is not a consensus difference on any corpus.

Other primary consensus verdicts (clean macro-F1):

- Vs anchor: different on all three (IEMOCAP +0.040, RAVDESS +0.048, CREMA-D +0.089). Selective is above the unimodal anchor.
- Vs logit average: IEMOCAP inconclusive (+0.012), RAVDESS inconclusive (−0.023), CREMA-D equivalent (−0.005).
- Vs QMF: IEMOCAP inconclusive (+0.015), RAVDESS inconclusive (−0.024), CREMA-D equivalent (+0.002).

## Secondary: selective Essence-AU vs always-on Essence-AU

S1 does not hold on any corpus (the 95% CI includes 0, or the estimate is not higher). S2 under §4.5 holds on all three (`s2_bootstrap` true: IEMOCAP +0.006, RAVDESS −0.006, CREMA-D −0.000, each 90% CI inside ±0.02, and each VRR 95% CI excludes 0). `s2_consensus` is false on all three.

Consensus, clean macro-F1: IEMOCAP inconclusive, RAVDESS inconclusive, CREMA-D equivalent. Zero-video VRR: IEMOCAP different (−0.070); RAVDESS and CREMA-D inconclusive. Vs anchor, clean macro-F1 is different on all three. Vs logit average and vs QMF, clean macro-F1 is inconclusive on IEMOCAP and RAVDESS and equivalent on CREMA-D.

## Cross-corpus

Do not present selective fusion as the paper's contribution. The primary comparison fails S1 and S2 on RAVDESS, where selective ConcatMatched-AU is worse than always-on fusion and worse than logit averaging and QMF in the point estimate. Where the primary bootstrap flags pass (IEMOCAP, CREMA-D), the clean gain is about +0.007 and the v2 consensus calls it equivalent to always-on ConcatMatched-AU. The VRR reduction, which is the other half of S2, is not consensus-different on any primary comparison. Selective fusion does beat the unimodal anchor on clean macro-F1 on all three corpora. It does not beat weighted logit averaging or QMF under the consensus rule on any corpus.
