# CML fidelity audit

Dated 8 October 2026, before any faithful-CML training result. Source: Ma et al., Calibrating Multimodal Learning, ICML 2023, arXiv:2306.01265 (HTML of 8 October 2026). Checked against Algorithm 1, Eq. (1)–(6), Section 3.1, Appendix B.5, Appendix C, Appendix H.2, and Appendix H.5. The review of 5 October 2026 (pages 3 and 5) asked for this check. The existing 45-run grid is left as the CML-inspired penalty and is not rerun.

## Published objective

Algorithm 1, for a classifier with M modalities:

1. Start from the full set S = M. Set L_CL to the classification loss on the full input, and L_CML to 0.
2. For m = 1 … M−1: randomly remove one modality from S and call the remainder T; add the classification loss on T to L_CL; add max(0, Conf(T) − Conf(S)) to L_CML; then set S ← T.
3. Minimise (1/M) L_CL + λ L_CML.

Section 3.1 defines Conf(x) = max_y P(y | θ, x), the maximum softmax probability. Eq. (4) is that hinge. Eq. (6) writes L_CL + λ L_CML; Algorithm 1 is the operational form and divides the summed classification losses by M. L_CL is a classification criterion. The paper’s example is cross-entropy. Nothing in Algorithm 1 detaches a confidence or a classification term. Appendix H.5 chooses λ on a validation set from {1, 5, 10, …, 100}. Appendix H.2 says VRR is a label-free internal ranking metric, distinct from ECE and Brier. Appendix B.5 remarks that, in a confidence plot, the regulariser was not added when the full-input prediction was wrong. That remark is not in Algorithm 1.

## Audio-visual mapping (M = 2)

ConcatMatched-AU has two modalities, audio (Wav2Vec2) and video (facial action units). The Algorithm 1 loop runs once. Each optimiser step draws one modality with equal probability and erases it. The draw uses a Python generator seeded with the training seed, so it does not consume the torch stream used by dropout or the data loader.

Erasure, for this architecture, is zero-fill: the erased modality’s feature values are replaced by zeros and its mask is left unchanged. That is the export condition `zero_audio` or `zero_video`. It is not `branch_disabled_audio` or `branch_disabled_video`, which zero the pooled encoder output and skip that encoder. It is not the Gaussian-noise stand-in the paper uses for MMTM, which cannot predict after a modality is deleted. ConcatMatched always receives both tensors, so zero-fill is the removal the exports already score.

With one sampled partial input the objective is

L = (1/2) (CE(full) + CE(partial)) + λ mean(relu(Conf(partial) − Conf(full))).

The hinge is the batch mean, the same reduction as cross-entropy in this code. It is not divided by M. Neither confidence is detached. On IEMOCAP the existing valence–activation concordance (weight 0.2) stays on the full input only and is not inside the 1/2. RAVDESS and CREMA-D have no valence–activation term, so their objective is the display above. λ = 1.0 is one fixed untuned setting. It was not selected on these development sets.

## Side-by-side

| Item | Published Algorithm 1 | Existing implementation (`cml_variant: penalty`, the 45-run grid) | Faithful implementation (`cml_variant: faithful`) |
| --- | --- | --- | --- |
| Classification loss | Cross-entropy on the full input and on every sampled partial input, then divided by M | Task loss on the full input only. IEMOCAP also includes valence–activation concordance at weight 0.2. No partial-input classification loss | (1/2) (CE(full) + CE(one partial)). IEMOCAP valence–activation concordance stays on the full input, outside the 1/2, as in the baseline and the penalty runs |
| Partial inputs | One random removal chain. For M = 2, one of the two modalities, redrawn each update | Both modalities, every batch. Not random | One of audio or video, uniform, each update. Seeded Python generator |
| Confidence | Max softmax probability | Max softmax probability | Max softmax probability. Not the true-class probability |
| Ranking term | Sum of hinges along the chain. For M = 2, one hinge max(0, Conf(partial) − Conf(full)). A tie is 0 | Sum of two batch-mean hinges, audio-removed and video-removed | One batch-mean hinge for the sampled partial. A tie is 0 |
| Normalisation | 1/M on the sum of classification losses only. The hinge sum is not divided by M | No 1/M. The penalty is the sum of the two means | 1/2 on the two cross-entropies. The hinge is not divided by 2 |
| Gradient | No stop-gradient in Algorithm 1. Full and partial classification losses and both confidences receive gradient | No stop-gradient. The full task loss and both removed confidences receive gradient. Partial inputs have no classification loss | No stop-gradient. Full and partial cross-entropies and both confidences receive gradient |
| Removal | “Randomly remove / erase” a modality. Imputation-independent models drop it. MMTM uses strong noise because it cannot drop a modality | Zero-fill features, mask unchanged | Same zero-fill, so training removal matches export `zero_audio` / `zero_video` |
| λ | Chosen on validation from {1, 5, …, 100} (Appendix H.5) | 1.0, fixed, not tuned here | 1.0, one fixed untuned setting. Not called the authors’ default |
| Appendix B.5 gate | Not part of Algorithm 1. A figure caption says the penalty was withheld when the full-input prediction was wrong | Not implemented | Not implemented. Algorithm 1 is followed |

The penalty variant remains the default. Configs that omit `training.cml_variant` keep it, and a zero `training.cml_weight` still adds nothing. The faithful configs are separate files and do not alter the penalty yaml files.
