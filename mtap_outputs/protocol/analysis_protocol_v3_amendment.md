# Amendment to analysis protocol (version 3)

Dated: 29 September 2026, 22:45 IST. Written after both new corpora were downloaded and verified, and **before any RAVDESS or CREMA-D feature was extracted or any model was trained on them**. No selective-fusion result exists yet. Versions 1 and 2 are unchanged.

## 1. Corpora and frozen splits

| Corpus | Clips | Classes (index) | Speakers | Five groups (stored as `session` 1-5) |
|---|---|---|---|---|
| RAVDESS | 672 full audio-video speech clips (modality 01, channel 01) | angry 05 -> 0, happy 03 -> 1, neutral 01 -> 2, sad 04 -> 3 | 24 actors | actors 01-05, 06-10, 11-15, 16-20, 21-24 |
| CREMA-D | 4,900 clips (ANG 1,271, HAP 1,271, NEU 1,087, SAD 1,271) | ANG -> 0, HAP -> 1, NEU -> 2, SAD -> 3 | 91 actors | see rule below |

CREMA-D grouping rule (deterministic, sex-balanced): from `VideoDemographics.csv`, sort male ActorIDs ascending and assign them to groups 1,2,3,4,5,1,2,... in order; do the same for female ActorIDs, continuing the cycle where the male list stopped. The resulting actor-to-group table is written to the split manifest and hashed before training.

For every corpus, fold k tests group k, develops on group (k mod 5) + 1, and trains on the other three groups. This reproduces Dr Singh's RAVDESS table exactly. These are proposed actor-disjoint splits, not official benchmarks. Class 1 is "happy" on both new corpora (IEMOCAP merges happy and excited).

Neither corpus has dimensional (valence/arousal) labels. The valence/arousal loss and metrics are **not computed** for them; they are not set to a placeholder value.

## 2. Systems trained on each new corpus (seeds 0, 1, 7; folds 1-5)

audio-only (Wav2Vec2), video-only (AU), video-only (ResNet-18), ConcatMatched-ResNet, Essence-ResNet, ConcatMatched-AU, Essence-AU. Seven systems x 15 = 105 runs per corpus. IEMOCAP additionally receives video-only (AU) at seeds 0, 1, 7 (15 runs) so the late-fusion baselines can be computed there.

Identical hyperparameters to the IEMOCAP configurations. No tuning on the new corpora. Visual features use the whole frame (single speaker); IEMOCAP keeps its speaker-half crop.

## 3. Selective fusion: primary variant fixed now

- **Anchor** per outer fold: the unimodal system (audio-only or video-only AU) with the higher seed-averaged development macro-F1.
- **Candidate fused system**: ConcatMatched-AU (primary); Essence-AU (secondary).
- **Gate inputs** per utterance, all computed from development-available quantities: anchor max probability, anchor entropy, fused max probability, fused entropy, argmax agreement (0/1), fused-minus-anchor max probability, and visual evidence (realized valid visual-token count divided by 32; 0 when video is absent or zero-filled).
- **Gate**: L2-regularised logistic regression (C = 1.0, standardised inputs) fitted on development utterances where exactly one of anchor and fused is correct; target 1 when the fused system is the correct one.
- **Threshold** tau: chosen on development data from {0.30, 0.35, ..., 0.70} to maximise development macro-F1 of the selective output; ties go to the larger tau (fewer switches).
- **Output**: fused probabilities when gate >= tau, otherwise the anchor's probabilities. All probabilities use each system's development-fitted temperature.
- **Condition behaviour**: under video interventions the same fitted gate is applied to the altered fused output and altered visual evidence; the anchor output is unchanged if the anchor is audio-only.
- Development data also served early stopping; this makes the gate mildly optimistic on development data but never touches test labels. Stated in the paper.

Post-hoc comparators on identical folds and seeds: anchor-only; always-on ConcatMatched-AU and Essence-AU; weighted logit averaging of audio-only and video-only AU (weight chosen on development from {0, 0.1, ..., 1}); QMF-style energy-weighted late fusion of the same two unimodal systems. CML-regularised ConcatMatched-AU is a trained comparator scheduled after the overnight grids.

Success criteria S1/S2 and the consensus verdict rules from versions 1-2 apply unchanged.
