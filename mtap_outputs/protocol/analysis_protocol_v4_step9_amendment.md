# Amendment to analysis protocol (version 4): Step 9 controls

Dated: 2 October 2026. Written after the CML comparator and before any Step 9 training, export, or test-set score. Versions 1, 2, and 3 are unchanged. `step9_decision.md` (1 October 2026) skipped these controls by narrowing the claims. That file is kept. This amendment replaces that decision: both controls are run.

No Step 9 test label has been seen. Widths below were chosen from parameter counts only.

## 1. What Step 9 requires, and what this run does

Dr Singh's Step 9 (26 September 2026 guide), not a substitute for the checkpoint audit:

- **9A.** If distillation is claimed to cause the gain or the robustness result, compare Essence with both directional losses on versus off. Architecture, initialisation seed, data order, optimiser, learning rate, epoch budget, Stage-2 eligibility, and the development selection rule stay the same. One-direction ablations are not required.
- **9B.** If total parameter capacity is claimed to explain the result, count trainable fusion and classification parameters separately from frozen extractors, fix a matching tolerance before test evaluation (the guide proposes 2% relative to the reference), choose width and depth from parameter counts under an equal tuning budget, and run that architecture on the same seeds and folds. Also measure inference latency with warm-up, device synchronisation, batch size, peak memory, and hardware.

Fully masked audio and video are a Step 6 export condition. Step 9 does not require them. This amendment does not add that condition and does not re-export existing checkpoints under a new name.

Corpora: IEMOCAP, RAVDESS, and CREMA-D. Seeds 0, 1, 7. Folds 1–5. Existing Essence-AU and ConcatMatched-AU checkpoints are the on-variant and the encoder-matched reference. Their protocols match this grid except for the single opt-in change named for each control (loss flag, or encoder depth and fusion width). They are reused, not retrained.

## 2. Runtime plan (from measured logs, not the old shell ETAs)

Mean completed-run times already on disk:

| Corpus | Essence-AU | ConcatMatched-AU | 30 new runs |
|---|---|---|---|
| RAVDESS | 128 s (`train_ravdess_progress.txt`) | about 64 s | about 0.8 h |
| CREMA-D | 910 s | 458 s | about 6.1 h |
| IEMOCAP | no Essence-AU log; CREMA-D Essence scaled by 5531/4900 | 716 s (CML ConcatMatched-AU) | about 7.3 h |

Export and the three speaker-level analyses add about 2 h. The total is about 16 h, under the 20 h cap, so all three corpora are in this protocol. RAVDESS is first because it is the short check; IEMOCAP and CREMA-D follow. One MPS job at a time. If the drive disconnects, the grid script skips complete metrics files, keeps failed metrics files, and deletes only a checkpoint that has no metrics file.

## 3. 9A Distillation off

`training.directional_distillation` defaults to true in `load_config`. Existing yaml distillation weights are then unchanged (both directional terms 0.02 in Stage 2; both 0 in Stage 1). Setting it false zeros `audio_to_video_distillation` and `video_to_audio_distillation` in every stage. That is the same change as `--disable-loss` on both names. The teacher modules stay in the architecture. Configs: `configs/{iemocap,ravdess,cremad}_w2v_video_au_nodist.yaml`, each the corpus AU yaml plus that one flag. System name `essence_au_nodist`. Model `essence`. Same Essence learning rate (0.0001), patience (15), epoch budget (50), Stage-2 rule, and development macro-F1 selection.

## 4. 9B Parameter budget

Frozen Wav2Vec2, ResNet-18, and OpenFace extractors are not in the training graph (0 trainable extractor parameters). Counts below are the trainable module at hidden size 256, dropout 0.2 (dropout does not add parameters), audio dimension 768.

| Variant | Video dim | Trainable | Projectors | Encoders | Fusion head | Classifier | Other |
|---|---:|---:|---:|---:|---:|---:|---:|
| ConcatMatched-AU | 20 | 3,495,726 | 203,816 | 3,159,040 (2+2 layers) | 131,328 | 1,542 | 0 |
| Essence-AU | 20 | 2,837,808 | 203,816 | 1,579,520 (fusion) | — | 1,542 | 1,052,930 |
| ConcatMatched-ResNet | 512 | 3,622,662 | 330,752 | 3,159,040 | 131,328 | 1,542 | 0 |
| Essence-ResNet | 512 | 2,964,744 | 330,752 | 1,579,520 | — | 1,542 | 1,052,930 |

Published ConcatMatched already exceeds Essence by 657,918 parameters (23.2% of Essence-AU; 22.2% of Essence-ResNet). The fusion head plus classifier is 132,870 parameters, so no fusion width at encoder depth 2 lands within 2% of Essence. The smallest head (width 1, depth 2) is still about 18.5% above Essence-AU.

Locked rule, fixed before any Step 9 test score: search `encoder_layers` in {1, 2} and integer `fusion_width` >= 1; take the pair with the smallest absolute gap to Essence; accept it only if that gap is at most 2% of the Essence count. No development or test metric is used. Equal tuning budget: one architecture, the published optimiser, learning rate, patience, and epoch budget.

Chosen control, for both video dimensions: `encoder_layers: 1`, `fusion_width: 2032`.

| Variant | Video dim | Trainable | Gap vs Essence | Relative to Essence |
|---|---:|---:|---:|---:|
| Capacity-matched Concat | 20 | 2,837,950 | +142 | 0.0050% |
| Capacity-matched Concat | 512 | 2,964,886 | +142 | 0.0048% |

The trained grid is the AU control (`concat_matched_au_budget`) on all three corpora. ResNet counts are reported and not trained. Configs: `configs/{iemocap,ravdess,cremad}_w2v_video_au_budget.yaml`. ConcatMatched uses learning rate 0.0003 and patience 8, as in the published runs.

Latency, measured once before the grid on untrained AU modules (dropout 0), not used to pick the width: MPS, batch size 16, 400 audio tokens, 32 video tokens, 10 warm-up iterations, 50 timed iterations, `torch.mps.synchronize()` before the timed window, peak allocated memory recorded. Output: `mtap_outputs/logs/step9_latency.json`.

## 5. Pairs, hypotheses, and verdicts

Effects are model B minus model A. Pair files, not to be edited after results:

- `pairs_step9_iemocap.json` (baseline name `concatmatched_au`)
- `pairs_step9_ravdess.json` and `pairs_step9_cremad.json` (baseline name `concat_matched_au`)

Families, locked now:

1. `step9_distillation`: Essence-AU vs Essence-AU nodist; ConcatMatched-AU vs Essence-AU nodist.
2. `step9_parameter_budget`: Essence-AU vs capacity-matched ConcatMatched-AU.

Hypotheses, two-sided, primary outcomes clean macro-F1, zero-audio VRR, zero-video VRR, and clean equal-width ECE:

- H1. Turning both directional distillation losses off changes Essence-AU on those outcomes.
- H2. Essence-AU with distillation off differs from ConcatMatched-AU on those outcomes.
- H3. The capacity-matched ConcatMatched-AU differs from Essence-AU on those outcomes.

Verdicts are protocol v2 consensus, unchanged: **different** only if the 95% speaker-bootstrap interval excludes 0 and the Holm-adjusted session-level paired t p is below 0.05; **equivalent** only if both 90% intervals (speaker bootstrap and session-level t) lie inside (−0.02, +0.02), and only for macro-F1, UAR, and accuracy; **inconclusive** otherwise. Equivalence is reported before difference when both apply. Holm uses the script's existing grouping (family, metric, condition, calibration). The family is not narrowed after the results. ECE and VRR are not equivalence-margin metrics. The resampling unit is the speaker (unstratified). Five sessions limit the paired t. Seeds are averaged within each held-out session before that test.

## 6. What a verdict allows the manuscript to say

- H1 equivalent on clean macro-F1: do not say distillation causes the Essence versus Concat clean-F1 gap.
- H1 different: distillation changes Essence under this protocol; report the sign and do not generalise beyond these corpora.
- H3 equivalent: do not say the remaining Essence versus Concat difference is total trainable capacity. H3 different: capacity-matching changes the comparison; report the sign. Either way, state that published ConcatMatched is encoder-matched and has more trainable parameters than Essence, and that the control shortened the encoders to one layer and widened the fusion head.
- Null and reversed results are reported. These controls do not establish a clinical or naturalistic claim.
