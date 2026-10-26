"""Selective fusion and post-hoc late fusion from analysis protocol v3, section 3.

Temperatures, the logistic gate and tau are fitted on development predictions
only. Test labels are not an input to fitting or threshold selection.

Ambiguities the protocol does not settle, and the rule used here:

- Equal seed-averaged development macro-F1: the audio-only system is the
  anchor when it is one of the tied systems; otherwise the lexicographic name.
- Entropy is the natural logarithm. Standardisation removes that scale.
- The standardiser is fit on the disagreement rows only.
- An empty or single-class disagreement set uses a constant gate probability
  (the empirical rate, or 0.5 if empty) and is recorded. It is not retuned.
- Logit-average weight ``w`` multiplies the audio-only temperature-scaled
  logits. Ties on development macro-F1 keep the larger ``w``.
- Late fusion averages temperature-scaled logits ``z / T``.
- QMF energy is ``-logsumexp(z / T)`` per modality. Weights are the softmax of
  the negative energies, so they are the softmax of ``logsumexp(z / T)`` and
  sum to one. Fused logits are the weighted sum of ``z / T``. No parameters
  are trained.
- Visual evidence is ``video_kept_tokens / 32``. ``zero_video`` and
  ``branch_disabled_video`` are 0 even when the mask still counts tokens.
- "Switched to fused" means the gate probability is at least tau.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_COMPANION = Path(__file__).resolve().parents[2] / "mer_companion"
if str(_COMPANION) not in sys.path:
    sys.path.insert(0, str(_COMPANION))

from merkit.analysis import classification, fit_temperature, softmax  # noqa: E402

VISUAL_TOKEN_BUDGET = 32
TAU_GRID = tuple(round(0.30 + 0.05 * i, 2) for i in range(9))
LOGIT_WEIGHT_GRID = tuple(round(0.1 * i, 1) for i in range(11))
VIDEO_ABSENT_CONDITIONS = frozenset({"zero_video", "branch_disabled_video"})
GATE_FEATURE_NAMES = (
    "anchor_max_probability",
    "anchor_entropy",
    "fused_max_probability",
    "fused_entropy",
    "argmax_agreement",
    "fused_minus_anchor_max_probability",
    "visual_evidence",
)
FUSED_TO_SELECTIVE = {
    "concatmatched_au": "selective_cmau",
    "essence_au": "selective_esau",
}
PREDICTION_COLUMNS = [
    "dataset", "model", "seed", "fold", "run_id", "utterance_id", "condition", "split", "label",
    *[f"p_{i}" for i in range(4)], *[f"z_{i}" for i in range(4)],
    "video_valid_tokens", "video_kept_tokens", "corruption_seed",
]
LATE_FUSION_SKIPPED_NOTE = (
    "skipped: video-only AU exports are absent, so weighted logit averaging and "
    "QMF-style fusion are not defined; the anchor is the available unimodal system"
)


class ConstantGate:
    """Gate used when the disagreement set cannot identify a logistic model."""

    def __init__(self, probability: float):
        self.probability_value = float(probability)

    def probability(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), self.probability_value, dtype=np.float64)


class ScaledLogisticGate:
    def __init__(self, scaler: StandardScaler, classifier: LogisticRegression):
        self.scaler = scaler
        self.classifier = classifier

    def probability(self, features: np.ndarray) -> np.ndarray:
        transformed = self.scaler.transform(np.asarray(features, dtype=np.float64))
        return self.classifier.predict_proba(transformed)[:, 1].astype(np.float64)


def is_audio_unimodal(name: str) -> bool:
    lowered = str(name).lower()
    if lowered in {"audio_w2v", "audio", "audio_only"}:
        return True
    blocked = ("concat", "essence", "fusion", "matched", "selective")
    return lowered.startswith("audio") and not any(token in lowered for token in blocked)


def is_video_au_unimodal(name: str) -> bool:
    lowered = str(name).lower()
    if lowered in {"video_au", "iemocap_video_au", "video_only_au"}:
        return True
    blocked = ("concat", "essence", "fusion", "matched", "selective", "resnet", "v2")
    return "video" in lowered and "au" in lowered and not any(token in lowered for token in blocked)


def is_unimodal_anchor_candidate(name: str) -> bool:
    return is_audio_unimodal(name) or is_video_au_unimodal(name)


def choose_anchor(seed_averaged_dev_macro_f1: dict[str, float]) -> str:
    """Pick the unimodal system with the higher seed-averaged development macro-F1."""
    if not seed_averaged_dev_macro_f1:
        raise ValueError("No unimodal system is available for anchor selection")
    best = max(float(value) for value in seed_averaged_dev_macro_f1.values())
    tied = [name for name, value in seed_averaged_dev_macro_f1.items() if float(value) == best]
    if len(tied) == 1:
        return tied[0]
    audio = sorted(name for name in tied if is_audio_unimodal(name))
    if audio:
        return audio[0]
    return sorted(tied)[0]


def shannon_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def visual_evidence(condition: str, kept_tokens, budget: int = VISUAL_TOKEN_BUDGET) -> np.ndarray:
    """Realised valid visual tokens over 32. Absent or zero-filled video is 0."""
    kept = np.asarray(kept_tokens, dtype=np.float64)
    if condition in VIDEO_ABSENT_CONDITIONS:
        return np.zeros(kept.shape, dtype=np.float64)
    if budget <= 0:
        raise ValueError("Visual token budget must be positive")
    if not np.isfinite(kept).all():
        raise ValueError(f"Non-finite visual token counts for condition {condition}")
    return kept / float(budget)


def gate_feature_matrix(anchor_p, fused_p, visual) -> np.ndarray:
    anchor_p = np.asarray(anchor_p, dtype=np.float64)
    fused_p = np.asarray(fused_p, dtype=np.float64)
    evidence = np.asarray(visual, dtype=np.float64)
    if anchor_p.shape != fused_p.shape or anchor_p.ndim != 2 or anchor_p.shape[1] != 4:
        raise ValueError("Anchor and fused probabilities must be Nx4")
    if evidence.shape != (len(anchor_p),):
        raise ValueError("Visual evidence must have one value per utterance")
    anchor_max = anchor_p.max(axis=1)
    fused_max = fused_p.max(axis=1)
    agreement = (anchor_p.argmax(axis=1) == fused_p.argmax(axis=1)).astype(np.float64)
    return np.column_stack([
        anchor_max,
        shannon_entropy(anchor_p),
        fused_max,
        shannon_entropy(fused_p),
        agreement,
        fused_max - anchor_max,
        evidence,
    ])


def fit_gate(features, labels, anchor_pred, fused_pred):
    """L2 logistic regression, C=1, on rows where exactly one system is correct.

    The target is 1 when the fused system is the correct one. Returns the gate,
    a note, and the number of fitting rows.
    """
    labels = np.asarray(labels, dtype=int)
    anchor_ok = np.asarray(anchor_pred, dtype=int) == labels
    fused_ok = np.asarray(fused_pred, dtype=int) == labels
    mask = anchor_ok != fused_ok
    target = fused_ok[mask].astype(np.int64)
    if int(mask.sum()) == 0 or len(np.unique(target)) < 2:
        value = float(target.mean()) if int(mask.sum()) else 0.5
        note = "disagreement set has a single class or is empty; constant gate probability"
        return ConstantGate(value), note, int(mask.sum())
    scaler = StandardScaler()
    training = scaler.fit_transform(np.asarray(features, dtype=np.float64)[mask])
    classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=0)
    classifier.fit(training, target)
    if np.any(np.asarray(classifier.n_iter_) >= 2000):
        raise RuntimeError("L2 logistic regression did not converge on the development disagreement set")
    return ScaledLogisticGate(scaler, classifier), "", int(mask.sum())


def select_tau(gate_probability, anchor_p, fused_p, labels) -> tuple[float, float]:
    """Development macro-F1 on TAU_GRID. Exact ties keep the larger tau."""
    labels = np.asarray(labels, dtype=int)
    gate_probability = np.asarray(gate_probability, dtype=np.float64)
    anchor_pred = np.asarray(anchor_p).argmax(axis=1)
    fused_pred = np.asarray(fused_p).argmax(axis=1)
    best_tau, best_f1 = None, None
    for tau in TAU_GRID:
        predicted = np.where(gate_probability >= tau, fused_pred, anchor_pred)
        score = classification(labels, predicted)["macro_f1"]
        if best_tau is None or score > best_f1 or (score == best_f1 and tau > best_tau):
            best_tau, best_f1 = float(tau), float(score)
    return best_tau, best_f1


def selective_logits(gate_probability, tau, anchor_z, fused_z):
    """Fused logits when the gate is at least tau, otherwise the anchor logits."""
    use_fused = np.asarray(gate_probability, dtype=np.float64) >= float(tau)
    chosen = np.array(anchor_z, dtype=np.float64, copy=True)
    chosen[use_fused] = np.asarray(fused_z, dtype=np.float64)[use_fused]
    return chosen, use_fused


def select_logit_weight(labels, audio_z, audio_temperature, video_z, video_temperature) -> tuple[float, float]:
    """Development weight on audio-only temperature-scaled logits. Ties take the larger weight."""
    labels = np.asarray(labels, dtype=int)
    best_weight, best_f1 = None, None
    for weight in LOGIT_WEIGHT_GRID:
        fused = averaged_logits(audio_z, audio_temperature, video_z, video_temperature, weight)
        score = classification(labels, softmax(fused).argmax(axis=1))["macro_f1"]
        if best_weight is None or score > best_f1 or (score == best_f1 and weight > best_weight):
            best_weight, best_f1 = float(weight), float(score)
    return best_weight, best_f1


def averaged_logits(audio_z, audio_temperature, video_z, video_temperature, audio_weight) -> np.ndarray:
    """``w * z_audio / T_audio + (1 - w) * z_video / T_video``."""
    weight = float(audio_weight)
    audio = np.asarray(audio_z, dtype=np.float64) / float(audio_temperature)
    video = np.asarray(video_z, dtype=np.float64) / float(video_temperature)
    return weight * audio + (1.0 - weight) * video


def qmf_fusion(audio_z, audio_temperature, video_z, video_temperature):
    """Energy-weighted late fusion. Weights sum to 1 and no parameters are fit.

    ``energy_m = -logsumexp(z_m / T_m)``.
    ``weight = softmax_m(-energy_m)``.
    ``z = sum_m weight_m * (z_m / T_m)``.
    """
    from scipy.special import logsumexp

    audio = np.asarray(audio_z, dtype=np.float64) / float(audio_temperature)
    video = np.asarray(video_z, dtype=np.float64) / float(video_temperature)
    energy = -np.column_stack([logsumexp(audio, axis=1), logsumexp(video, axis=1)])
    negative = -energy
    shifted = negative - negative.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    weights /= weights.sum(axis=1, keepdims=True)
    fused = weights[:, 0:1] * audio + weights[:, 1:2] * video
    return fused, weights


def selective_outputs(dev: dict, test: dict) -> dict:
    """Fit on development arrays and apply to test arrays.

    ``test`` may contain ``y``. That entry is ignored, so shuffling test labels
    cannot change the returned probabilities.
    """
    dev_labels = np.asarray(dev["y"], dtype=int)
    anchor_temperature, _anchor_info = fit_temperature(dev_labels, dev["anchor_z"])
    fused_temperature, _fused_info = fit_temperature(dev_labels, dev["fused_z"])
    anchor_dev = softmax(np.asarray(dev["anchor_z"], dtype=np.float64) / anchor_temperature)
    fused_dev = softmax(np.asarray(dev["fused_z"], dtype=np.float64) / fused_temperature)
    dev_features = gate_feature_matrix(anchor_dev, fused_dev, dev["visual"])
    gate, note, n_fit = fit_gate(dev_features, dev_labels, anchor_dev.argmax(1), fused_dev.argmax(1))
    dev_gate = gate.probability(dev_features)
    tau, dev_f1 = select_tau(dev_gate, anchor_dev, fused_dev, dev_labels)
    anchor_test = np.asarray(test["anchor_z"], dtype=np.float64) / anchor_temperature
    fused_test = np.asarray(test["fused_z"], dtype=np.float64) / fused_temperature
    anchor_p = softmax(anchor_test)
    fused_p = softmax(fused_test)
    test_features = gate_feature_matrix(anchor_p, fused_p, test["visual"])
    test_gate = gate.probability(test_features)
    chosen, switched = selective_logits(test_gate, tau, anchor_test, fused_test)
    dev_chosen, dev_switched = selective_logits(
        dev_gate, tau,
        np.asarray(dev["anchor_z"], dtype=np.float64) / anchor_temperature,
        np.asarray(dev["fused_z"], dtype=np.float64) / fused_temperature,
    )
    return {
        "logits": chosen,
        "probabilities": softmax(chosen),
        "tau": tau,
        "dev_macro_f1": dev_f1,
        "coverage_test": float(switched.mean()) if len(switched) else float("nan"),
        "coverage_dev": float(dev_switched.mean()) if len(dev_switched) else float("nan"),
        "anchor_temperature": float(anchor_temperature),
        "fused_temperature": float(fused_temperature),
        "n_fit": n_fit,
        "gate_note": note,
        "dev_logits": dev_chosen,
    }
