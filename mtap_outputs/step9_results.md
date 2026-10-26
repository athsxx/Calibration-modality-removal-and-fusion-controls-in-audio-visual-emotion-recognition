# Step 9 results (2 October 2026)

The grid finished at 2026-10-02 23:19 (`mtap_outputs/logs/step9_status.md`). Training, export, and speaker-level analysis completed with zero failed runs. This file is the write-up. `step9_decision.md` (1 October 2026) is kept as the earlier decision to skip the controls. Protocol version 4 replaced that decision before any Step 9 test score.

Analyses: `mtap_outputs/analysis_step9_{iemocap,ravdess,cremad}/`. Effects are model B minus model A. Consensus is protocol v2: **different** only if the speaker-bootstrap 95% interval excludes 0 and the Holm-adjusted session-level paired *t* *p* is below 0.05; **equivalent** only if both 90% intervals lie inside ±0.02, and only for macro-F1, UAR, and accuracy; **inconclusive** otherwise. Equivalence is reported before difference when both apply. ECE and VRR are not equivalence-margin metrics. Bootstrap *B* = 10,000, hierarchical sensitivity *B* = 2,000, unstratified speaker resampling (`strong_analysis_manifest.json`).

Locked pairs (`protocol/pairs_step9_*.json`):

1. `essence_au` → `essence_au_nodist` (family `step9_distillation`)
2. `concatmatched_au` (IEMOCAP) or `concat_matched_au` (RAVDESS, CREMA-D) → `essence_au_nodist`
3. `essence_au` → `concat_matched_au_budget` (family `step9_parameter_budget`)

A capacity-matched versus published ConcatMatched-AU pair was not locked. No consensus verdict is given for it. Fully masked audio and video were not added. The version 4 amendment says they are a Step 6 export condition, and the exporter still lists `fully_masked_audio` and `fully_masked_video` as unimplemented.

Primary rows below are raw probabilities. Clean ECE is `ece_width`. Temperature-scaled ECE is reported because the manuscript's calibration claim uses that scaling, and one raw ECE verdict changes under it.

## Trainable parameters

Frozen Wav2Vec2, ResNet-18, and OpenFace extractors are outside the training graph. Counts are the trainable module at hidden size 256 and action-unit video dimension 20. Dropout adds no parameters. The three timed architectures are `mtap_outputs/logs/step9_latency.json` (`models.*.trainable_parameters`), which matches the counts locked in `protocol/analysis_protocol_v4_step9_amendment.md` before any test score. The same totals were reproduced by constructing `ConcatFusionModelMatched` and `EssenceTransferModel` on CPU.

| Variant | Trainable | Where recorded |
|---|---:|---|
| ConcatMatched-AU (2+2 encoder layers) | 3,495,726 | latency JSON; protocol table |
| Essence-AU (fusion depth 2) | 2,837,808 | latency JSON; protocol table |
| Essence-AU, distillation off | 2,837,808 | same Essence module; the flag zeros both distillation losses and adds no parameters. Checkpoint `artifacts/checkpoints/essence_fold_1_iemocap_essence_au_nodist_seed0.pt` state-dict numel is 2,837,808 |
| ConcatMatched-AU budget (`encoder_layers` 1, `fusion_width` 2032) | 2,837,950 | latency JSON; protocol table. Checkpoint `artifacts/checkpoints/concat_matched_fold_1_iemocap_concat_matched_au_budget_seed0.pt` state-dict numel is 2,837,950 |

The budget gap versus Essence is +142 parameters, 0.0050% of the Essence count. Published ConcatMatched-AU exceeds Essence by 657,918 parameters. The protocol records the split (projectors, encoders, fusion head, classifier, other) and states that no encoder depth of 2 lands inside the 2% tolerance, which is why the control shortens the encoders to one layer and widens the fusion head. ResNet parameter counts in that table were not trained.

## Latency

Untrained action-unit modules, dropout disabled, measured once before the grid (`step9_latency.json`). Device MPS, batch size 16, 400 audio tokens, 32 video tokens, audio dimension 768, video dimension 20, 10 warm-up iterations, 50 timed iterations, `torch.mps.synchronize()` before the timed window. Platform macOS-27.2 arm64, Python 3.11.8, torch 2.13.0.

| Model | Seconds per batch | Peak allocated bytes |
|---|---:|---:|
| ConcatMatched-AU | 0.005739624179841485 | 33,692,416 |
| Essence-AU | 0.009244363339967095 | 47,328,768 |
| ConcatMatched-AU budget | 0.0034617208197596483 | 62,900,224 |

Displays used in the manuscript, four decimal places for seconds: 0.0057, 0.0092, 0.0035. Essence is the slowest of the three. The wider one-layer head is the fastest and has the highest peak allocation. Nodist was not timed; it is the Essence forward graph.

**Superseded (26 October 2026).** This profiling output is unsuitable for peak-memory comparisons. The "Peak allocated bytes" column is current MPS tensor allocation, not peak memory, and the three models were allocated in one process, so allocations accumulated across rows. The timings are from untrained modules. The 13 October revised manuscript removed both the memory comparison and the timing paragraph; no rerun is needed for its claims.

## Consensus verdicts

Order inside each slash is IEMOCAP / RAVDESS / CREMA-D. Row numbers are identical across the three `paired_comparisons.csv` files. CSV row 1 is the header.

### Distillation off minus Essence-AU

| Outcome | Effect | Consensus | Row |
|---|---|---|---:|
| Clean macro-F1 | −0.0019 / +0.0031 / −0.0035 | equivalent / equivalent / equivalent | 30 |
| Zero-video VRR | −0.0114 / −0.0382 / +0.0025 | inconclusive / inconclusive / inconclusive | 104 |
| Zero-audio VRR | +0.0007 / −0.0055 / −0.0028 | inconclusive / inconclusive / inconclusive | 90 |
| Clean ECE, raw | −0.0040 / −0.0029 / +0.0015 | inconclusive / inconclusive / inconclusive | 35 |
| Clean ECE, temperature | −0.0014 / +0.0001 / +0.0026 | inconclusive / inconclusive / inconclusive | 41 |

### Distillation off minus ConcatMatched-AU

| Outcome | Effect | Consensus | Row |
|---|---|---|---:|
| Clean macro-F1 | +0.0004 / +0.0256 / +0.0032 | inconclusive / inconclusive / equivalent | 140 |
| Zero-video VRR | +0.0451 / +0.0590 / +0.0757 | inconclusive / inconclusive / different | 214 |
| Zero-audio VRR | −0.0095 / +0.0451 / +0.0573 | inconclusive / inconclusive / different | 200 |
| Clean ECE, raw | +0.0750 / −0.0082 / +0.0156 | inconclusive / inconclusive / inconclusive | 145 |
| Clean ECE, temperature | +0.0063 / +0.0056 / +0.0017 | inconclusive / inconclusive / inconclusive | 151 |

On CREMA-D the two VRR differences stay different after temperature scaling: zero-audio +0.0641 (row 207) and zero-video +0.0861 (row 221).

### Capacity-matched ConcatMatched-AU minus Essence-AU

| Outcome | Effect | Consensus | Row |
|---|---|---|---:|
| Clean macro-F1 | −0.0059 / −0.0271 / −0.0099 | inconclusive / inconclusive / equivalent | 250 |
| Zero-video VRR | −0.0488 / −0.1131 / −0.0639 | different / different / different | 324 |
| Zero-audio VRR | −0.0604 / −0.0699 / −0.0452 | inconclusive / different / inconclusive | 310 |
| Clean ECE, raw | −0.1145 / +0.0096 / −0.0048 | different / inconclusive / inconclusive | 255 |
| Clean ECE, temperature | −0.0142 / +0.0095 / +0.0030 | inconclusive / inconclusive / inconclusive | 261 |

Temperature-scaled zero-video VRR stays different: −0.0466 / −0.1151 / −0.0703 (row 331). The RAVDESS zero-audio contrast is inconclusive after temperature scaling: −0.0511 (row 317, Holm *p* = 0.0597).

Speaker-pooled clean raw macro-F1 (`point_and_ci_metrics.csv`, column `estimate`): distillation-off 0.6111 / 0.7922 / 0.8047 (row 522); budget 0.6071 / 0.7620 / 0.7983 (row 42). These levels are not all printed in the manuscript. The printed claims use the paired effects.

## What the manuscript may claim

Distillation does not cause the Essence versus ConcatMatched clean macro-F1 gap. Turning both directional losses off leaves Essence equivalent on clean macro-F1 on IEMOCAP, RAVDESS, and CREMA-D. The same within-Essence contrasts on zero-audio VRR, zero-video VRR, and clean ECE are inconclusive, so distillation is not identified as the cause of those outcomes either. The action-unit gain over audio is unchanged as a separate result and is not attributed to distillation.

On CREMA-D, distillation-off Essence has a higher zero-audio VRR and a higher zero-video VRR than ConcatMatched-AU (consensus different). That is a between-architecture contrast. The within-Essence distillation contrast on those rates is inconclusive, so the manuscript does not say the loss produced the gap.

Trainable parameter count does not explain the Essence versus ConcatMatched clean macro-F1 comparison. The capacity-matched ConcatMatched is equivalent to Essence on CREMA-D clean macro-F1 and inconclusive on IEMOCAP and RAVDESS. Published ConcatMatched stays the encoder-matched reference and has more trainable parameters than Essence. The control that matched the count shortened the encoders to one layer and widened the fusion head, because no depth-2 width fell inside the pre-registered 2% tolerance.

Capacity matching does change zero-video VRR. The capacity-matched model is lower than Essence by a consensus difference on all three corpora. On RAVDESS it is also lower on raw zero-audio VRR; that contrast is inconclusive after temperature scaling. On IEMOCAP, raw clean ECE is lower for the capacity-matched model (consensus different); temperature-scaled clean ECE is inconclusive for every locked contrast, including that one. These signs are reported. They are not a claim that total parameter count produced the clean scores.

These controls do not establish a clinical or naturalistic claim.
