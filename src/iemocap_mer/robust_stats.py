"""Speaker-cluster inference for exported multimodal emotion-recognition logits.

The primary resampling unit is the speaker. Out-of-fold test rows are pooled
across folds before a metric is computed, then averaged across seeds inside
each replicate. Equal-mass ECE and AURC are not linear in speaker weights.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import digamma, logsumexp
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_estimating_equations import GEE

N_CLASSES = 4
N_BINS = 15
BOOT_SEED = 20260929
EQUIVALENCE_MARGIN = 0.02
LOSS_FRACTIONS = (0.0, 0.25, 0.5, 0.75)
LOSS_CONDITIONS = ("clean", "video_token_loss_25", "video_token_loss_50", "video_token_loss_75")
# np.digitize edges matching merkit.analysis.score (15 equal-width bins).
BIN_EDGES = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
SCALAR_METRICS = (
    "macro_f1", "uar", "accuracy", "nll", "brier",
    "ece_width", "mce_width", "ece_mass", "aurc", "classwise_ece",
)
PAIR_METRICS = ("macro_f1", "uar", "accuracy", "nll", "brier", "ece_width")
EFFECT_METRICS = (
    "vrr",
    "mean_clean_minus_altered_confidence",
    "clean_correct_altered_wrong_fraction",
    "clean_wrong_altered_correct_fraction",
)
NONADDITIVE = ("ece_mass", "aurc")
FAMILY_DEFAULT = "iemocap_historical_posthoc"
LIMITATIONS = (
    "only 10 IEMOCAP speakers; bootstrap CIs approximate",
    "IEMOCAP family post hoc",
    "Hierarchical bootstrap is a sensitivity analysis; primary intervals resample speakers.",
    "Equal-mass ECE and AURC are not linear in speaker weights; their intervals are hierarchical only.",
    "Negative method-of-moments variance components are truncated at 0.",
    "GEE stacks seeds inside each speaker cluster with an exchangeable working correlation.",
)


def _companion():
    root = Path(__file__).resolve().parents[2] / "mer_companion"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from merkit.analysis import (
        classification,
        fit_temperature,
        holm,
        paired,
        score,
        softmax,
        validate_table,
    )
    from merkit.common import FOLDS, LABELS, PRIMARY_PAIRS, SEEDS, read_json, sha256, timestamp

    return {
        "classification": classification,
        "fit_temperature": fit_temperature,
        "holm": holm,
        "paired": paired,
        "score": score,
        "softmax": softmax,
        "validate_table": validate_table,
        "FOLDS": FOLDS,
        "LABELS": LABELS,
        "PRIMARY_PAIRS": PRIMARY_PAIRS,
        "SEEDS": SEEDS,
        "read_json": read_json,
        "sha256": sha256,
        "timestamp": timestamp,
    }


C = _companion()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def bin_index(confidence: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(confidence, dtype=np.float64), BIN_EDGES).astype(np.int64)


def utterance_parts(y, z, temperature: float = 1.0) -> dict:
    """Per-utterance terms. NLL uses logsumexp so extreme logits are not clipped."""
    y = np.asarray(y, dtype=np.int64)
    effective = np.asarray(z, dtype=np.float64) / float(temperature)
    p = C["softmax"](effective)
    pred = p.argmax(1).astype(np.int64)
    conf = p.max(1)
    correct = (pred == y).astype(np.float64)
    nll = logsumexp(effective, axis=1) - effective[np.arange(len(y)), y]
    brier = ((p - np.eye(N_CLASSES)[y]) ** 2).sum(1)
    return {
        "y": y,
        "p": p,
        "pred": pred,
        "conf": conf,
        "correct": correct,
        "nll": nll,
        "brier": brier,
        "bin_id": bin_index(conf),
        "effective_z": effective,
    }


def confusion_matrix(y, pred) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    return np.bincount(N_CLASSES * y + pred, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES).astype(np.float64)


def metrics_from_confusion(cm: np.ndarray) -> dict:
    """Macro-F1, UAR, accuracy and per-class scores from a (possibly weighted) 4x4 matrix."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diagonal(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros(N_CLASSES), where=denom > 0)
    recall = np.divide(tp, support, out=np.zeros(N_CLASSES), where=support > 0)
    n = float(cm.sum())
    accuracy = float(tp.sum() / n) if n else float("nan")
    return {
        "macro_f1": float(f1.mean()),
        "uar": float(recall.mean()),
        "accuracy": accuracy,
        "per_class_f1": f1,
        "per_class_recall": recall,
        "n": n,
    }


def weighted_confusion(cms: np.ndarray, multiplicities: np.ndarray) -> np.ndarray:
    """cms is (speakers, 4, 4); multiplicities is (speakers,)."""
    cms = np.asarray(cms, dtype=np.float64)
    multiplicities = np.asarray(multiplicities, dtype=np.float64)
    return np.tensordot(multiplicities, cms, axes=(0, 0))


def _batch_classification(cm: np.ndarray) -> dict:
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support = cm.sum(axis=-1)
    predicted = cm.sum(axis=-2)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(denom), where=denom > 0)
    recall = np.divide(tp, support, out=np.zeros_like(support), where=support > 0)
    n = cm.sum(axis=(-2, -1))
    accuracy = np.divide(tp.sum(axis=-1), n, out=np.zeros(n.shape), where=n > 0)
    return {
        "macro_f1": f1.mean(axis=-1),
        "uar": recall.mean(axis=-1),
        "accuracy": accuracy,
        "per_class_f1": f1,
        "per_class_recall": recall,
        "n": n,
    }


def ece_width_from_bins(bin_count, bin_conf, bin_correct, n) -> tuple[np.ndarray, np.ndarray]:
    occupied = bin_count > 0
    acc = np.divide(bin_correct, bin_count, out=np.zeros_like(bin_count), where=occupied)
    cf = np.divide(bin_conf, bin_count, out=np.zeros_like(bin_conf), where=occupied)
    gaps = np.abs(acc - cf)
    ece = np.divide((bin_count * gaps).sum(axis=-1), n, out=np.full(np.shape(n), np.nan), where=n > 0)
    masked = np.where(occupied, gaps, -np.inf)
    mce = masked.max(axis=-1)
    mce = np.where(np.isfinite(mce), mce, np.nan)
    return ece, mce


def classwise_ece_from_bins(count, conf_sum, correct_sum, n) -> np.ndarray:
    """Mean of per-class equal-width ECEs. Class k uses confidence p_k and label indicator y=k."""
    occupied = count > 0
    acc = np.divide(correct_sum, count, out=np.zeros_like(count), where=occupied)
    cf = np.divide(conf_sum, count, out=np.zeros_like(conf_sum), where=occupied)
    gaps = np.abs(acc - cf)
    ece_k = np.divide(
        (count * gaps).sum(axis=-1),
        np.expand_dims(n, -1),
        out=np.zeros(count.shape[:-1]),
        where=np.expand_dims(n, -1) > 0,
    )
    return ece_k.mean(axis=-1)


def _mass_edges(totals: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Expanded-index edges. Chunk sizes follow numpy.array_split (leading chunks are larger)."""
    totals = np.asarray(totals, dtype=np.int64)
    k = np.minimum(n_bins, np.maximum(totals, 0))
    safe = np.maximum(k, 1)
    neach = np.floor_divide(totals, safe)
    extras = np.mod(totals, safe)
    idx = np.arange(n_bins + 1)
    i = idx[None, :]
    neach_b = neach[:, None]
    extras_b = extras[:, None]
    k_b = k[:, None]
    edges = np.where(i <= extras_b, i * (neach_b + 1), extras_b * (neach_b + 1) + (i - extras_b) * neach_b)
    edges = np.where(i <= k_b, edges, totals[:, None])
    edges[:, 0] = 0
    return edges.astype(np.float64)


def equal_mass_ece(conf, correct, weights=None, n_bins: int = N_BINS) -> np.ndarray | float:
    """Equal-mass ECE. Unit weights match merkit.analysis.score via array_split chunk sizes."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    scalar = weights is None
    if weights is None:
        weights = np.ones((1, len(conf)), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.ndim == 1:
            weights = weights[None, :]
    order = np.argsort(conf, kind="stable")
    conf_s = conf[order]
    cor_s = correct[order]
    w = weights[:, order]
    totals = w.sum(axis=1)
    edges = _mass_edges(np.rint(totals).astype(np.int64), n_bins)
    c1 = np.cumsum(w, axis=1)
    c0 = c1 - w
    sum_n = np.zeros((w.shape[0], n_bins))
    sum_conf = np.zeros_like(sum_n)
    sum_cor = np.zeros_like(sum_n)
    bin_left = edges[:, :-1]
    bin_right = edges[:, 1:]
    for j in range(w.shape[1]):
        if not np.any(w[:, j] > 0):
            continue
        overlap = np.maximum(0.0, np.minimum(c1[:, j:j + 1], bin_right) - np.maximum(c0[:, j:j + 1], bin_left))
        sum_n += overlap
        sum_conf += overlap * conf_s[j]
        sum_cor += overlap * cor_s[j]
    occupied = sum_n > 0
    acc = np.divide(sum_cor, sum_n, out=np.zeros_like(sum_n), where=occupied)
    cf = np.divide(sum_conf, sum_n, out=np.zeros_like(sum_n), where=occupied)
    ece = np.divide((sum_n * np.abs(acc - cf)).sum(axis=1), totals, out=np.full(len(totals), np.nan), where=totals > 0)
    return float(ece[0]) if scalar else ece


def _harmonic(n: np.ndarray) -> np.ndarray:
    # H_0 = 0. digamma(n+1) + γ equals H_n for positive integers.
    return digamma(np.asarray(n, dtype=np.float64) + 1.0) + np.euler_gamma


def aurc(conf, correct, weights=None) -> np.ndarray | float:
    """Mean selective risk of confidence-ordered prefixes (descending confidence).

    Integer weights expand an utterance before the prefix average. The unit-weight
    path is the mean of cumulative error rates and does not go through digamma.
    """
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    order = np.argsort(-conf, kind="stable")
    err = 1.0 - correct[order]
    if weights is None:
        risk = np.cumsum(err) / np.arange(1, len(err) + 1)
        return float(risk.mean()) if len(err) else float("nan")
    w = np.asarray(weights, dtype=np.float64)
    scalar = w.ndim == 1
    if scalar:
        w = w[None, :]
    w = w[:, order]
    totals = w.sum(axis=1)
    acc = np.zeros(w.shape[0])
    seen = np.zeros(w.shape[0])
    errors = np.zeros(w.shape[0])
    for j in range(w.shape[1]):
        wi = w[:, j]
        if not np.any(wi):
            continue
        e = float(err[j])
        acc += e * wi + (errors - e * seen) * (_harmonic(seen + wi) - _harmonic(seen))
        errors += e * wi
        seen += wi
    out = np.divide(acc, totals, out=np.full(len(totals), np.nan), where=totals > 0)
    return float(out[0]) if scalar else out


def accumulate_speakers(parts: dict, speaker_index: np.ndarray, n_speakers: int) -> dict:
    speaker_index = np.asarray(speaker_index, dtype=np.int64)
    s = n_speakers
    cm = np.zeros((s, N_CLASSES, N_CLASSES), dtype=np.float64)
    np.add.at(cm, (speaker_index, parts["y"], parts["pred"]), 1.0)
    count = np.bincount(speaker_index, minlength=s).astype(np.float64)
    nll_sum = np.bincount(speaker_index, weights=parts["nll"], minlength=s)
    brier_sum = np.bincount(speaker_index, weights=parts["brier"], minlength=s)
    flat = speaker_index * N_BINS + parts["bin_id"]
    bin_count = np.bincount(flat, minlength=s * N_BINS).reshape(s, N_BINS).astype(np.float64)
    bin_conf = np.bincount(flat, weights=parts["conf"], minlength=s * N_BINS).reshape(s, N_BINS)
    bin_correct = np.bincount(flat, weights=parts["correct"], minlength=s * N_BINS).reshape(s, N_BINS)
    class_count = np.zeros((s, N_CLASSES, N_BINS))
    class_conf = np.zeros_like(class_count)
    class_correct = np.zeros_like(class_count)
    for k in range(N_CLASSES):
        conf_k = parts["p"][:, k]
        bid = bin_index(conf_k)
        flat_k = speaker_index * N_BINS + bid
        class_count[:, k] = np.bincount(flat_k, minlength=s * N_BINS).reshape(s, N_BINS)
        class_conf[:, k] = np.bincount(flat_k, weights=conf_k, minlength=s * N_BINS).reshape(s, N_BINS)
        class_correct[:, k] = np.bincount(
            flat_k, weights=(parts["y"] == k).astype(np.float64), minlength=s * N_BINS
        ).reshape(s, N_BINS)
    return {
        "cm": cm,
        "nll_sum": nll_sum,
        "brier_sum": brier_sum,
        "count": count,
        "bin_count": bin_count,
        "bin_conf": bin_conf,
        "bin_correct": bin_correct,
        "class_count": class_count,
        "class_conf": class_conf,
        "class_correct": class_correct,
    }


def metrics_from_speaker_stats(stats: dict, multiplicities: np.ndarray | None = None) -> dict:
    """Seed-averaged metrics. stats arrays are (seeds, speakers, ...). multiplicities is (B, speakers)."""
    cm = stats["cm"]
    if multiplicities is None:
        cm_w = cm.sum(axis=1)
        nll_w = stats["nll_sum"].sum(axis=1)
        brier_w = stats["brier_sum"].sum(axis=1)
        count_w = stats["count"].sum(axis=1)
        bin_count = stats["bin_count"].sum(axis=1)
        bin_conf = stats["bin_conf"].sum(axis=1)
        bin_correct = stats["bin_correct"].sum(axis=1)
        class_count = stats["class_count"].sum(axis=1)
        class_conf = stats["class_conf"].sum(axis=1)
        class_correct = stats["class_correct"].sum(axis=1)
        seed_axis = 0
    else:
        mult = np.asarray(multiplicities, dtype=np.float64)
        cm_w = np.tensordot(mult, cm, axes=([1], [1]))
        nll_w = np.tensordot(mult, stats["nll_sum"], axes=([1], [1]))
        brier_w = np.tensordot(mult, stats["brier_sum"], axes=([1], [1]))
        count_w = np.tensordot(mult, stats["count"], axes=([1], [1]))
        bin_count = np.tensordot(mult, stats["bin_count"], axes=([1], [1]))
        bin_conf = np.tensordot(mult, stats["bin_conf"], axes=([1], [1]))
        bin_correct = np.tensordot(mult, stats["bin_correct"], axes=([1], [1]))
        class_count = np.tensordot(mult, stats["class_count"], axes=([1], [1]))
        class_conf = np.tensordot(mult, stats["class_conf"], axes=([1], [1]))
        class_correct = np.tensordot(mult, stats["class_correct"], axes=([1], [1]))
        seed_axis = 1
    cls = _batch_classification(cm_w)
    ece, mce = ece_width_from_bins(bin_count, bin_conf, bin_correct, count_w)
    out = {
        "macro_f1": cls["macro_f1"],
        "uar": cls["uar"],
        "accuracy": cls["accuracy"],
        "nll": np.divide(nll_w, count_w, out=np.full(count_w.shape, np.nan), where=count_w > 0),
        "brier": np.divide(brier_w, count_w, out=np.full(count_w.shape, np.nan), where=count_w > 0),
        "ece_width": ece,
        "mce_width": mce,
        "classwise_ece": classwise_ece_from_bins(class_count, class_conf, class_correct, count_w),
        "per_class_f1": cls["per_class_f1"],
        "per_class_recall": cls["per_class_recall"],
    }
    return {key: np.mean(value, axis=seed_axis) for key, value in out.items()}


def metrics_unit_weight(y, z, temperature: float = 1.0) -> dict:
    """Single-matrix metrics through the speaker-statistic path (one speaker, multiplicity 1)."""
    parts = utterance_parts(y, z, temperature)
    stats = accumulate_speakers(parts, np.zeros(len(parts["y"]), dtype=np.int64), 1)
    stacked = {key: value[None, ...] for key, value in stats.items()}
    reduced = metrics_from_speaker_stats(stacked, None)
    reduced["ece_mass"] = equal_mass_ece(parts["conf"], parts["correct"])
    reduced["aurc"] = aurc(parts["conf"], parts["correct"])
    return reduced


def condition_effect_counts(y, z_clean, z_altered, temperature: float = 1.0) -> dict:
    """Strict VRR: max(p_altered) > max(p_clean). Ties are not violations."""
    clean = utterance_parts(y, z_clean, temperature)
    altered = utterance_parts(y, z_altered, temperature)
    violations = altered["conf"] > clean["conf"]
    cc_aw = (clean["correct"] == 1) & (altered["correct"] == 0)
    cw_ac = (clean["correct"] == 0) & (altered["correct"] == 1)
    n = int(len(y))
    return {
        "n": n,
        "violations": int(violations.sum()),
        "conf_gap_sum": float((clean["conf"] - altered["conf"]).sum()),
        "cc_aw": int(cc_aw.sum()),
        "cw_ac": int(cw_ac.sum()),
        "vrr": float(violations.mean()) if n else float("nan"),
        "mean_clean_minus_altered_confidence": float((clean["conf"] - altered["conf"]).mean()) if n else float("nan"),
        "clean_correct_altered_wrong_fraction": float(cc_aw.mean()) if n else float("nan"),
        "clean_wrong_altered_correct_fraction": float(cw_ac.mean()) if n else float("nan"),
    }


def speaker_multiplicities(strata: np.ndarray, n_boot: int, seed: int = BOOT_SEED,
                           mode: str = "unstratified") -> np.ndarray:
    """Speaker multiplicities of shape (n_boot, n_speakers).

    ``unstratified`` draws the full speaker set with replacement (primary, protocol v2).
    ``stratified`` redraws inside each stratum and keeps that stratum's speaker count.
    The generator is seeded internally, so the same call is deterministic.
    """
    if mode not in ("unstratified", "stratified"):
        raise ValueError("resampling mode must be 'unstratified' or 'stratified'")
    strata = np.asarray(strata)
    n_speakers = len(strata)
    rng = np.random.default_rng(seed)
    if mode == "unstratified":
        draws = rng.choice(n_speakers, size=(n_boot, n_speakers), replace=True)
        offset = np.arange(n_boot)[:, None] * n_speakers
        flat = (draws + offset).ravel()
        return np.bincount(flat, minlength=n_boot * n_speakers).reshape(n_boot, n_speakers)
    mult = np.zeros((n_boot, n_speakers), dtype=np.int64)
    levels = sorted(set(strata.tolist()), key=lambda value: (str(type(value)), str(value)))
    for level in levels:
        idx = np.flatnonzero(strata == level)
        k = len(idx)
        draws = rng.choice(idx, size=(n_boot, k), replace=True)
        offset = np.arange(n_boot)[:, None] * n_speakers
        flat = (draws + offset).ravel()
        mult += np.bincount(flat, minlength=n_boot * n_speakers).reshape(n_boot, n_speakers)
    return mult


def bootstrap_pvalue(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=np.float64)
    b = len(differences)
    if b == 0:
        return float("nan")
    p = 2.0 * min(float(np.mean(differences <= 0)), float(np.mean(differences >= 0)))
    return float(max(1.0 / b, min(1.0, p)))


def percentile_intervals(samples: np.ndarray) -> dict:
    samples = np.asarray(samples, dtype=np.float64)
    low95, high95 = np.percentile(samples, [2.5, 97.5])
    low90, high90 = np.percentile(samples, [5.0, 95.0])
    return {
        "ci95_low": float(low95),
        "ci95_high": float(high95),
        "ci90_low": float(low90),
        "ci90_high": float(high90),
    }


MARGIN_METRICS = ("macro_f1", "uar", "accuracy")


def _inside_margin(low, high, margin: float = EQUIVALENCE_MARGIN) -> bool:
    return bool(np.isfinite(low) and np.isfinite(high) and low > -margin and high < margin)


def _excludes_zero(low, high) -> bool:
    return bool(np.isfinite(low) and np.isfinite(high) and (low > 0.0 or high < 0.0))


def equivalence_verdict(ci90_low, ci90_high, ci95_low, ci95_high, margin: float = EQUIVALENCE_MARGIN) -> str:
    """90% CI strictly inside (-margin, margin), else 'different' if the 95% CI excludes 0."""
    if _inside_margin(ci90_low, ci90_high, margin):
        return "equivalent"
    if _excludes_zero(ci95_low, ci95_high):
        return "different"
    return "inconclusive"


def bootstrap_component_verdict(ci90_low, ci90_high, ci95_low, ci95_high, metric: str,
                                margin: float = EQUIVALENCE_MARGIN) -> str:
    """Speaker-bootstrap component. Equivalence is available only for macro-F1, UAR and accuracy."""
    if metric in MARGIN_METRICS and _inside_margin(ci90_low, ci90_high, margin):
        return "equivalent"
    if _excludes_zero(ci95_low, ci95_high):
        return "different"
    return "inconclusive"


def fold_component_verdict(ci90_low, ci90_high, holm_p, metric: str, margin: float = EQUIVALENCE_MARGIN) -> str:
    """Fold-level component. 'different' uses the Holm-adjusted paired t p."""
    if metric in MARGIN_METRICS and _inside_margin(ci90_low, ci90_high, margin):
        return "equivalent"
    if np.isfinite(holm_p) and float(holm_p) < 0.05:
        return "different"
    return "inconclusive"


def consensus_verdict(ci90_low, ci90_high, ci95_low, ci95_high, fold_ci90_low, fold_ci90_high, holm_p,
                      metric: str, margin: float = EQUIVALENCE_MARGIN) -> str:
    """Both procedures must agree. Equivalence, when both 90% intervals sit inside the margin, is reported first."""
    different = _excludes_zero(ci95_low, ci95_high) and np.isfinite(holm_p) and float(holm_p) < 0.05
    if metric not in MARGIN_METRICS:
        return "different" if different else "inconclusive"
    equivalent = _inside_margin(ci90_low, ci90_high, margin) and _inside_margin(fold_ci90_low, fold_ci90_high, margin)
    if equivalent:
        return "equivalent"
    if different:
        return "different"
    return "inconclusive"


def macro_f1_contrast(y, pred_a, pred_b, speaker_index, strata, n_boot: int = 200, seed: int = BOOT_SEED,
                      mode: str = "unstratified") -> dict:
    """Single-seed macro-F1 contrast. Both models share labels, speakers and multiplicity draws."""
    y = np.asarray(y, dtype=np.int64)
    pred_a = np.asarray(pred_a, dtype=np.int64)
    pred_b = np.asarray(pred_b, dtype=np.int64)
    speaker_index = np.asarray(speaker_index, dtype=np.int64)
    n_speakers = int(speaker_index.max()) + 1
    dummy_z = None

    def _stats(pred):
        parts = {
            "y": y,
            "pred": pred,
            "p": np.eye(N_CLASSES)[pred],
            "conf": np.ones(len(y)),
            "correct": (pred == y).astype(np.float64),
            "nll": np.zeros(len(y)),
            "brier": np.zeros(len(y)),
            "bin_id": np.zeros(len(y), dtype=np.int64),
        }
        one = accumulate_speakers(parts, speaker_index, n_speakers)
        return {key: value[None, ...] for key, value in one.items()}

    del dummy_z
    stats_a = _stats(pred_a)
    stats_b = _stats(pred_b)
    mult = speaker_multiplicities(np.asarray(strata), n_boot, seed, mode=mode)
    samples = metrics_from_speaker_stats(stats_b, mult)["macro_f1"] - metrics_from_speaker_stats(stats_a, mult)["macro_f1"]
    point = float(
        metrics_from_speaker_stats(stats_b)["macro_f1"] - metrics_from_speaker_stats(stats_a)["macro_f1"]
    )
    intervals = percentile_intervals(samples)
    return {
        "effect": point,
        "samples": samples,
        "mult": mult,
        "p": bootstrap_pvalue(samples),
        "verdict": equivalence_verdict(intervals["ci90_low"], intervals["ci90_high"], intervals["ci95_low"], intervals["ci95_high"]),
        **intervals,
    }


def dersimonian_laird(effects, standard_errors) -> dict:
    """Random-effects pool. i2 is (Q - df) / Q on [0, 1], not a percentage."""
    y = np.asarray(effects, dtype=np.float64)
    se = np.asarray(standard_errors, dtype=np.float64)
    if y.shape != se.shape or y.ndim != 1 or len(y) < 1:
        raise ValueError("effects and standard_errors must be equal-length 1-d arrays")
    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(se)) or np.any(se <= 0):
        raise ValueError("effects must be finite and standard errors must be positive")
    k = len(y)
    w = 1.0 / se ** 2
    pooled_fe = float(np.sum(w * y) / np.sum(w))
    q = float(np.sum(w * (y - pooled_fe) ** 2))
    df = k - 1
    c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    if df <= 0 or c <= 0:
        tau2 = 0.0
        i2 = 0.0
    else:
        tau2 = max(0.0, (q - df) / c)
        i2 = max(0.0, (q - df) / q) if q > 0 else 0.0
    w_star = 1.0 / (se ** 2 + tau2)
    pooled = float(np.sum(w_star * y) / np.sum(w_star))
    se_pool = float(np.sqrt(1.0 / np.sum(w_star)))
    z = float(stats.norm.ppf(0.975))
    return {
        "pooled": pooled,
        "se": se_pool,
        "ci95_low": pooled - z * se_pool,
        "ci95_high": pooled + z * se_pool,
        "tau2": float(tau2),
        "i2": float(i2),
        "q": q,
        "k": k,
    }


def variance_components(fold_macro: pd.DataFrame) -> pd.DataFrame:
    """Two-way ANOVA without replication. Negative components are truncated at 0."""
    rows = []
    for model, group in fold_macro.groupby("model", sort=True):
        pivot = group.pivot(index="fold", columns="seed", values="macro_f1")
        note = ""
        truncated = ""
        if pivot.shape[0] < 2 or pivot.shape[1] < 2 or pivot.isna().any().any():
            rows.append({
                "model": model, "n_folds": int(pivot.shape[0]), "n_seeds": int(pivot.shape[1]),
                "ss_fold": np.nan, "ss_seed": np.nan, "ss_residual": np.nan,
                "var_fold": np.nan, "var_seed": np.nan, "var_residual": np.nan,
                "negative_truncated": "",
                "note": "incomplete seed-by-fold grid; variance components not estimated",
            })
            continue
        values = pivot.to_numpy(dtype=np.float64)
        a, b = values.shape
        grand = values.mean()
        fold_mean = values.mean(axis=1, keepdims=True)
        seed_mean = values.mean(axis=0, keepdims=True)
        ss_fold = float(b * np.sum((fold_mean - grand) ** 2))
        ss_seed = float(a * np.sum((seed_mean - grand) ** 2))
        ss_e = float(np.sum((values - fold_mean - seed_mean + grand) ** 2))
        ms_f, ms_s, ms_e = ss_fold / (a - 1), ss_seed / (b - 1), ss_e / ((a - 1) * (b - 1))
        components = {"fold": (ms_f - ms_e) / b, "seed": (ms_s - ms_e) / a, "residual": ms_e}
        truncated_names = []
        for name in ("fold", "seed", "residual"):
            if components[name] < 0:
                components[name] = 0.0
                truncated_names.append(name)
        if truncated_names:
            note = "negative method-of-moments component truncated at 0"
            truncated = ",".join(truncated_names)
        rows.append({
            "model": model, "n_folds": a, "n_seeds": b,
            "ss_fold": ss_fold, "ss_seed": ss_seed, "ss_residual": ss_e,
            "var_fold": float(components["fold"]), "var_seed": float(components["seed"]),
            "var_residual": float(components["residual"]),
            "negative_truncated": truncated, "note": note,
        })
    return pd.DataFrame(rows)


def paired_t_test(differences: np.ndarray) -> dict:
    diff = np.asarray(differences, dtype=np.float64)
    n = len(diff)
    if n == 0:
        return {"effect": float("nan"), "sd": float("nan"), "p": float("nan"),
                "ci95_low": float("nan"), "ci95_high": float("nan"),
                "ci90_low": float("nan"), "ci90_high": float("nan"), "df": 0, "note": "no folds"}
    effect = float(diff.mean())
    if n < 2:
        return {"effect": effect, "sd": float("nan"), "p": float("nan"),
                "ci95_low": float("nan"), "ci95_high": float("nan"),
                "ci90_low": float("nan"), "ci90_high": float("nan"), "df": n - 1,
                "note": "fewer than 2 folds"}
    sd = float(diff.std(ddof=1))
    if sd > 0:
        t_value = effect / (sd / np.sqrt(n))
        p = float(2 * stats.t.sf(abs(t_value), n - 1))
        half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
        half90 = float(stats.t.ppf(0.95, n - 1) * sd / np.sqrt(n))
        note = ""
    else:
        p = 1.0 if effect == 0 else float("nan")
        half = 0.0
        half90 = 0.0
        note = "zero variance in paired differences"
    return {"effect": effect, "sd": sd, "p": p, "ci95_low": effect - half, "ci95_high": effect + half,
            "ci90_low": effect - half90, "ci90_high": effect + half90, "df": n - 1, "note": note}


def minimum_detectable_effect(differences: np.ndarray) -> dict:
    """MDE = (t_0.975,df + t_0.80,df) * sd / sqrt(n) for a paired t-test."""
    diff = np.asarray(differences, dtype=np.float64)
    n = len(diff)
    if n < 2:
        return {"n": n, "sd": float("nan"), "mde": float("nan"), "note": "fewer than 2 folds"}
    sd = float(diff.std(ddof=1))
    df = n - 1
    mde = float((stats.t.ppf(0.975, df) + stats.t.ppf(0.80, df)) * sd / np.sqrt(n))
    return {"n": n, "sd": sd, "mde": mde, "note": "" if n == 5 else f"n={n} folds in this export, not 5"}


def _odds_ratio(fitted) -> dict:
    coef = float(np.asarray(fitted.params, dtype=float)[1])
    se = float(np.asarray(fitted.bse, dtype=float)[1])
    z = float(stats.norm.ppf(0.975))
    return {
        "or": float(np.exp(coef)),
        "ci_low": float(np.exp(coef - z * se)),
        "ci_high": float(np.exp(coef + z * se)),
        "p": float(np.asarray(fitted.pvalues, dtype=float)[1]),
    }


def fit_gee(frame: pd.DataFrame, model_a: str, model_b: str) -> dict:
    """Binomial GEE of correct ~ C(model). Primary covariance is Mancl-DeRouen bias-reduced."""
    empty = {
        "gee_or": float("nan"), "gee_ci95_low": float("nan"), "gee_ci95_high": float("nan"),
        "gee_p": float("nan"), "gee_error": "", "gee_covariance": "",
        "gee_robust_or": float("nan"), "gee_robust_p": float("nan"),
    }
    data = frame.loc[frame["model"].isin([model_a, model_b]), ["speaker_id", "model", "correct"]].copy()
    data["correct"] = data["correct"].astype(float)
    if data["speaker_id"].nunique() < 2 or set(data["model"]) != {model_a, model_b}:
        empty["gee_error"] = "GEE needs both models and at least two speakers"
        return empty
    try:
        # Treatment coding of C(model) with model_a as the reference. Built without
        # the formula API because statsmodels.api does not import on this SciPy build.
        endog = data["correct"].to_numpy(dtype=np.float64)
        exog = np.column_stack([
            np.ones(len(data)),
            (data["model"].to_numpy() == model_b).astype(np.float64),
        ])
        groups = data["speaker_id"].to_numpy()
        model = GEE(endog, exog, groups, family=Binomial(), cov_struct=Exchangeable())
        robust = model.fit(maxiter=100, cov_type="robust")
        if not bool(robust.converged):
            empty["gee_error"] = "GEE did not converge"
            return empty
        sandwich = _odds_ratio(robust)
        empty["gee_robust_or"] = sandwich["or"]
        empty["gee_robust_p"] = sandwich["p"]
        try:
            reduced = model.fit(maxiter=100, cov_type="bias_reduced", start_params=np.asarray(robust.params, dtype=float))
            if not bool(reduced.converged):
                raise RuntimeError("bias_reduced GEE did not converge")
            primary = _odds_ratio(reduced)
            empty["gee_covariance"] = "bias_reduced"
        except Exception as exc:
            primary = sandwich
            empty["gee_covariance"] = "robust"
            empty["gee_error"] = (
                f"bias_reduced failed, fell back to robust: {type(exc).__name__}: {exc}"
            ).splitlines()[0][:300]
        empty.update({
            "gee_or": primary["or"],
            "gee_ci95_low": primary["ci_low"],
            "gee_ci95_high": primary["ci_high"],
            "gee_p": primary["p"],
        })
        return empty
    except Exception as exc:
        empty["gee_error"] = f"{type(exc).__name__}: {exc}".splitlines()[0][:300]
        return empty


def _slope(values_at_fractions: np.ndarray) -> np.ndarray:
    """Least-squares slope against LOSS_FRACTIONS. Last axis is the four fractions."""
    x = np.asarray(LOSS_FRACTIONS, dtype=np.float64)
    xc = x - x.mean()
    values = np.asarray(values_at_fractions, dtype=np.float64)
    return np.tensordot(values, xc, axes=(-1, 0)) / np.dot(xc, xc)


def is_partial_manifest(manifest: dict) -> bool:
    if not manifest.get("complete", False):
        return True
    expected = int(manifest.get("expected_principal_count", 75))
    return int(manifest.get("selected_count", 0)) != expected


def load_pairs(pairs_json: str | Path | None) -> list[tuple[str, str, str]]:
    if pairs_json is None:
        return [(a, b, FAMILY_DEFAULT) for a, b in C["PRIMARY_PAIRS"]]
    payload = json.loads(Path(pairs_json).read_text(encoding="utf-8"))
    pairs = []
    for item in payload:
        if isinstance(item, dict):
            pairs.append((item["model_a"], item["model_b"], item.get("family", "custom")))
        else:
            pairs.append((item[0], item[1], item[2] if len(item) > 2 else "custom"))
    return pairs


@dataclass
class Block:
    model: str
    condition: str
    calibration: str
    seeds: tuple
    speaker_ids: tuple
    strata: np.ndarray
    utterance_id: np.ndarray
    speaker_index: np.ndarray
    folds: tuple
    stats: dict
    effect_stats: dict | None
    seeds_utterances: list
    point: dict
    per_class_point: dict
    partial: bool


def _stack_stats(per_seed: list[dict]) -> dict:
    return {key: np.stack([item[key] for item in per_seed], axis=0) for key in per_seed[0]}


def _logit_frame(ids, parts, model, seed, condition) -> pd.DataFrame:
    frame = pd.DataFrame({
        "model": model,
        "seed": int(seed),
        "fold": 0,
        "utterance_id": np.asarray(ids).astype(str),
        "condition": condition,
        "split": "test",
        "label": parts["y"].astype(int),
    })
    for k in range(N_CLASSES):
        frame[f"p_{k}"] = parts["p"][:, k]
        frame[f"z_{k}"] = parts["effective_z"][:, k]
    return frame


def _effect_arrays(clean: dict, altered: dict) -> dict:
    # Strict inequality: an unchanged maximum is not a violation.
    return {
        "violations": (altered["conf"] > clean["conf"]).astype(np.float64),
        "conf_gap": clean["conf"] - altered["conf"],
        "cc_aw": ((clean["correct"] == 1.0) & (altered["correct"] == 0.0)).astype(np.float64),
        "cw_ac": ((clean["correct"] == 0.0) & (altered["correct"] == 1.0)).astype(np.float64),
    }


def accumulate_effects(clean: dict, altered: dict, speaker_index: np.ndarray, n_speakers: int) -> dict:
    speaker_index = np.asarray(speaker_index, dtype=np.int64)
    arrays = _effect_arrays(clean, altered)
    out = {key: np.bincount(speaker_index, weights=value, minlength=n_speakers) for key, value in arrays.items()}
    out["count"] = np.bincount(speaker_index, minlength=n_speakers).astype(np.float64)
    out.update({f"u_{key}": value for key, value in arrays.items()})
    return out


def effect_rates(stats: dict, multiplicities: np.ndarray | None = None) -> dict:
    """Seed-averaged condition effects. stats arrays are (seeds, speakers) plus optional utterance vectors."""
    keys = ("violations", "conf_gap", "cc_aw", "cw_ac")
    names = {
        "violations": "vrr",
        "conf_gap": "mean_clean_minus_altered_confidence",
        "cc_aw": "clean_correct_altered_wrong_fraction",
        "cw_ac": "clean_wrong_altered_correct_fraction",
    }
    if multiplicities is None:
        den = stats["count"].sum(axis=1)
        num = {key: stats[key].sum(axis=1) for key in keys}
        seed_axis = 0
    else:
        mult = np.asarray(multiplicities, dtype=np.float64)
        den = np.tensordot(mult, stats["count"], axes=([1], [1]))
        num = {key: np.tensordot(mult, stats[key], axes=([1], [1])) for key in keys}
        seed_axis = 1
    out = {}
    for key, name in names.items():
        ratio = np.divide(num[key], den, out=np.full(np.shape(den), np.nan), where=den > 0)
        out[name] = np.mean(ratio, axis=seed_axis)
    return out


def speaker_effect_rates(y, z_clean, z_altered, speaker_index, temperature: float = 1.0) -> dict:
    """Additive speaker path for VRR and correctness transitions, summed back to corpus rates."""
    clean = utterance_parts(y, z_clean, temperature)
    altered = utterance_parts(y, z_altered, temperature)
    speaker_index = np.asarray(speaker_index, dtype=np.int64)
    n_speakers = int(speaker_index.max()) + 1
    one = accumulate_effects(clean, altered, speaker_index, n_speakers)
    stacked = {key: value[None, ...] for key, value in one.items() if not key.startswith("u_")}
    rates = effect_rates(stacked, None)
    return {key: float(value) for key, value in rates.items()} | {
        "n": int(len(y)),
        "violations": int(one["violations"].sum()),
    }


def metrics_from_weights(utt: dict, weights: np.ndarray) -> dict:
    """Utterance-weighted metrics for one seed. weights is (replicates, utterances)."""
    w = np.asarray(weights, dtype=np.float64)
    y = utt["y"]
    pred = utt["pred"]
    code = y * N_CLASSES + pred
    flat = np.zeros((len(w), N_CLASSES * N_CLASSES))
    for cls in range(N_CLASSES * N_CLASSES):
        mask = code == cls
        if np.any(mask):
            flat[:, cls] = w[:, mask].sum(axis=1)
    cm = flat.reshape(len(w), N_CLASSES, N_CLASSES)
    classified = _batch_classification(cm)
    count = w.sum(axis=1)
    oh = np.eye(N_BINS)[utt["bin_id"]]
    bin_count = w @ oh
    bin_conf = w @ (oh * utt["conf"][:, None])
    bin_correct = w @ (oh * utt["correct"][:, None])
    ece, mce = ece_width_from_bins(bin_count, bin_conf, bin_correct, count)
    class_count = np.zeros((len(w), N_CLASSES, N_BINS))
    class_conf = np.zeros_like(class_count)
    class_correct = np.zeros_like(class_count)
    for k in range(N_CLASSES):
        bid = bin_index(utt["p"][:, k])
        ohk = np.eye(N_BINS)[bid]
        class_count[:, k] = w @ ohk
        class_conf[:, k] = w @ (ohk * utt["p"][:, k:k + 1])
        class_correct[:, k] = w @ (ohk * (y == k).astype(np.float64)[:, None])
    return {
        "macro_f1": classified["macro_f1"],
        "uar": classified["uar"],
        "accuracy": classified["accuracy"],
        "nll": np.divide(w @ utt["nll"], count, out=np.full(len(w), np.nan), where=count > 0),
        "brier": np.divide(w @ utt["brier"], count, out=np.full(len(w), np.nan), where=count > 0),
        "ece_width": ece,
        "mce_width": mce,
        "classwise_ece": classwise_ece_from_bins(class_count, class_conf, class_correct, count),
        "ece_mass": equal_mass_ece(utt["conf"], utt["correct"], w),
        "aurc": aurc(utt["conf"], utt["correct"], w),
        "per_class_f1": classified["per_class_f1"],
        "per_class_recall": classified["per_class_recall"],
    }


def hierarchical_weights(speaker_index: np.ndarray, n_speakers: int, multiplicities: np.ndarray, rng) -> np.ndarray:
    """Within each drawn speaker, multinomial utterance weights of size n_speaker * multiplicity.

    A speaker drawn m times contributes m independent resamples; their sum is one multinomial
    of m * n_speaker draws.
    """
    speaker_index = np.asarray(speaker_index, dtype=np.int64)
    b, _ = multiplicities.shape
    weights = np.zeros((b, len(speaker_index)), dtype=np.int64)
    for speaker in range(n_speakers):
        idx = np.flatnonzero(speaker_index == speaker)
        n_speaker = len(idx)
        if n_speaker == 0:
            continue
        probability = np.full(n_speaker, 1.0 / n_speaker)
        trials = multiplicities[:, speaker].astype(np.int64) * n_speaker
        for replicate in np.flatnonzero(trials):
            weights[replicate, idx] = rng.multinomial(int(trials[replicate]), probability)
    return weights


def _mean_dicts(rows: list[dict], keys: tuple[str, ...]) -> dict:
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


_SCORE_KEYS = ("macro_f1", "uar", "accuracy", "nll", "brier", "ece_width", "ece_mass", "mce_width")
_CHECK_KEYS = ("macro_f1", "uar", "accuracy", "nll", "brier", "ece_width", "mce_width")


def _concat_parts(chunks: list[dict]) -> dict:
    keys = ("y", "p", "pred", "conf", "correct", "nll", "brier", "bin_id", "effective_z", "utterance_id")
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in keys}


def _temperature_lookup(path):
    if path is None:
        return None
    frame = pd.read_csv(path)
    required = {"model", "seed", "fold", "temperature"}
    if not required.issubset(frame.columns):
        raise ValueError("temperatures CSV must include model, seed, fold, temperature")
    lookup = {}
    for row in frame.itertuples(index=False):
        lookup[(str(row.model), int(row.seed), int(row.fold))] = float(row.temperature)
    return lookup


def _load_run_frames(exports: Path, manifest: dict, hashes: dict) -> list[tuple[dict, dict]]:
    loaded = []
    seen = set()
    root = exports.resolve()
    for run in manifest["runs"]:
        key = (run["model"], int(run["seed"]), int(run["fold"]))
        if key in seen:
            raise ValueError(f"Duplicate run in export manifest: {key}")
        seen.add(key)
        frames = {}
        for part in ("validation", "test"):
            record = run["files"][part]
            relative = f"runs/{run['run_id']}/{record['file']}"
            path = (exports / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Path escapes the export root: {relative}")
            digest = C["sha256"](path)
            if digest != record["sha256"]:
                raise ValueError(f"Prediction file changed since export: {relative}")
            hashes[relative] = digest
            frame = pd.read_csv(path, dtype={"utterance_id": str})
            if len(frame) != int(record["rows"]):
                raise ValueError(f"Prediction row count disagrees with export manifest: {relative}")
            for column in ("model", "seed", "fold"):
                if set(frame[column].map(str)) != {str(run[column])}:
                    raise ValueError(f"Prediction run identity disagrees with export manifest: {relative}")
            frames[part] = frame
        loaded.append((run, frames))
    if not loaded:
        raise ValueError("Export manifest contains no runs")
    return loaded


def _speaker_axis(ids: np.ndarray, meta: pd.DataFrame, strata_column: str):
    sub = meta.loc[list(ids)]
    stratum = sub.groupby("speaker_id")[strata_column].agg(lambda values: values.iloc[0])
    if int((sub.groupby("speaker_id")[strata_column].nunique() > 1).sum()):
        raise ValueError("A speaker maps to more than one stratum")
    # Manifest speaker ids may be integers (RAVDESS, CREMA-D). Lookups use strings.
    stratum.index = stratum.index.map(str)
    speaker_ids = tuple(sorted(stratum.index, key=lambda speaker: (stratum.loc[speaker], speaker)))
    index = {speaker: i for i, speaker in enumerate(speaker_ids)}
    strata = np.array([stratum.loc[speaker] for speaker in speaker_ids])
    speaker_index = np.array([index[str(meta.loc[uid, "speaker_id"])] for uid in ids], dtype=np.int64)
    return speaker_ids, strata, speaker_index


@dataclass
class Prepared:
    manifest: dict
    hashes: dict
    blocks: dict
    fold_scores: pd.DataFrame
    partial: bool
    temperature_source: str
    strata_column: str
    skipped_pairs: list


def prepare_export(exports, temperatures=None, strata_column: str = "session", allow_partial: bool = False) -> Prepared:
    exports = Path(exports).resolve()
    manifest = C["read_json"](exports / "export_manifest.json")
    if is_partial_manifest(manifest) and not allow_partial:
        raise ValueError("Partial export; pass --allow-partial for diagnostics only.")
    hashes = {"export_manifest.json": C["sha256"](exports / "export_manifest.json")}
    split_path = exports / "split_manifest.csv"
    hashes["split_manifest.csv"] = C["sha256"](split_path)
    expected_split = (manifest.get("settings") or {}).get("split_manifest_sha256")
    if expected_split and hashes["split_manifest.csv"] != expected_split:
        raise ValueError("Split manifest changed after export")
    split = pd.read_csv(split_path, dtype={"utterance_id": str})
    if strata_column not in split.columns:
        raise ValueError(f"Strata column {strata_column!r} is not in the split manifest")
    test_meta = split[split["split"] == "test"].copy()
    if test_meta["utterance_id"].duplicated().any():
        raise ValueError("Split manifest has duplicate test utterance ids")
    test_meta = test_meta.set_index("utterance_id", drop=False)
    lookup = _temperature_lookup(temperatures)
    if temperatures is not None:
        hashes["temperatures.csv"] = C["sha256"](temperatures)
    temperature_source = "temperatures_csv" if lookup is not None else "fit_on_clean_development"
    fold_rows = []
    run_store = {}
    for run, frames in _load_run_frames(exports, manifest, hashes):
        model, seed, fold = run["model"], int(run["seed"]), int(run["fold"])
        dev, test = frames["validation"], frames["test"]
        if set(dev["condition"]) != {"clean"}:
            raise ValueError("Temperature must be fitted on clean development rows only")
        y_dev, _, z_dev = C["validate_table"](dev, "validation")
        C["validate_table"](test, "test")
        if set(dev["utterance_id"]) & set(test["utterance_id"]):
            raise ValueError("Development/test utterance leakage")
        identity = (model, seed, fold)
        if lookup is None:
            temperature, _info = C["fit_temperature"](y_dev, z_dev)
        else:
            if identity not in lookup:
                raise ValueError(f"Missing temperature for {identity}")
            temperature = lookup[identity]
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"Temperature must be positive and finite for {identity}")
        expected_ids = set(test_meta.index[test_meta["fold"].to_numpy() == fold])
        clean = test[test["condition"] == "clean"]
        conditions = {}
        for condition, group in test.groupby("condition", sort=True):
            group = group.sort_values("utterance_id", kind="mergesort")
            y, _, z = C["validate_table"](group, "test")
            ids = group["utterance_id"].astype(str).to_numpy()
            if len(ids) != len(set(ids)) or set(ids) != expected_ids:
                raise ValueError(f"Test ids for {identity} {condition} disagree with the split manifest")
            if not np.array_equal(test_meta.loc[ids, "label"].to_numpy(), y):
                raise ValueError("Prediction labels disagree with the split manifest")
            if not np.isfinite(z).all():
                raise ValueError(f"Non-finite logits for {identity} {condition}")
            if not np.array_equal(C["softmax"](z).argmax(1), C["softmax"](z / temperature).argmax(1)):
                raise ValueError("Numerical argmax change after temperature scaling")
            conditions[condition] = {"y": y, "z": np.asarray(z, dtype=np.float64), "utterance_id": ids}
            for calibration, used in (("raw", 1.0), ("temperature", float(temperature))):
                scored = C["score"](y, z, used)
                row = {
                    "model": model, "seed": seed, "fold": fold, "condition": condition,
                    "calibration": calibration, "history_status": run.get("history_status", ""),
                    **{key: scored[key] for key in _SCORE_KEYS},
                }
                if condition != "clean":
                    compared = C["paired"](clean, group, used)
                    row.update({key: compared[key] for key in EFFECT_METRICS})
                fold_rows.append(row)
        if "clean" not in conditions:
            raise ValueError(f"Run {identity} has no clean condition")
        run_store[identity] = {"temperature": float(temperature), "conditions": conditions, "run_id": run["run_id"]}

    by_model_seed: dict[tuple, list] = {}
    for model, seed, fold in run_store:
        by_model_seed.setdefault((model, seed), []).append(fold)
    blocks = {}
    models = sorted({model for model, _, _ in run_store})
    for model in models:
        seeds = tuple(sorted(seed for owned, seed in by_model_seed if owned == model))
        coverages = []
        id_sets = []
        for seed in seeds:
            folds = tuple(sorted(by_model_seed[(model, seed)]))
            ids = []
            for fold in folds:
                ids.extend(run_store[(model, seed, fold)]["conditions"]["clean"]["utterance_id"].tolist())
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate out-of-fold utterance for {model} seed {seed}")
            coverages.append(folds)
            id_sets.append(tuple(sorted(ids)))
        if len(set(coverages)) != 1 or len(set(id_sets)) != 1:
            raise ValueError(f"Seeds of {model} do not share an out-of-fold utterance set")
        folds = coverages[0]
        ids = np.asarray(id_sets[0], dtype=str)
        expected = set(test_meta.index[test_meta["fold"].isin(folds)])
        if set(ids) != expected:
            raise ValueError(f"Out-of-fold ids for {model} do not match the split manifest")
        if not allow_partial and (set(folds) != set(C["FOLDS"]) or set(seeds) != set(C["SEEDS"])):
            raise ValueError(f"Incomplete fold/seed coverage for {model}")
        speaker_ids, strata, speaker_index = _speaker_axis(ids, test_meta, strata_column)
        n_speakers = len(speaker_ids)
        partial = set(folds) != set(C["FOLDS"]) or set(seeds) != set(C["SEEDS"]) or is_partial_manifest(manifest)
        condition_sets = []
        for seed in seeds:
            for fold in folds:
                condition_sets.append(set(run_store[(model, seed, fold)]["conditions"]))
        conditions = sorted(set.intersection(*condition_sets))
        pooled = {}
        for condition in conditions:
            for calibration in ("raw", "temperature"):
                per_seed = []
                for seed in seeds:
                    chunks = []
                    for fold in folds:
                        payload = run_store[(model, seed, fold)]
                        item = payload["conditions"][condition]
                        used = 1.0 if calibration == "raw" else payload["temperature"]
                        parts = utterance_parts(item["y"], item["z"], used)
                        parts["utterance_id"] = item["utterance_id"]
                        chunks.append(parts)
                    merged = _concat_parts(chunks)
                    order = np.argsort(merged["utterance_id"], kind="mergesort")
                    merged = {key: value[order] for key, value in merged.items()}
                    if not np.array_equal(merged["utterance_id"], ids):
                        raise ValueError(f"Out-of-fold alignment failed for {model} seed {seed} {condition}")
                    per_seed.append(merged)
                stats = _stack_stats([accumulate_speakers(parts, speaker_index, n_speakers) for parts in per_seed])
                reduced = metrics_from_speaker_stats(stats)
                scored = [C["score"](parts["y"], parts["effective_z"], 1.0) for parts in per_seed]
                point = _mean_dicts(scored, _SCORE_KEYS)
                for key in _CHECK_KEYS:
                    if abs(float(reduced[key]) - point[key]) > 1e-8:
                        raise ValueError(f"Speaker statistics disagree with score() for {model} {condition} {key}")
                for parts, one in zip(per_seed, scored):
                    mass = float(equal_mass_ece(parts["conf"], parts["correct"]))
                    if abs(mass - one["ece_mass"]) > 1e-8:
                        raise ValueError(f"Equal-mass ECE disagrees with score() for {model} {condition}")
                point["ece_mass"] = float(np.mean([one["ece_mass"] for one in scored]))
                point["aurc"] = float(np.mean([aurc(parts["conf"], parts["correct"]) for parts in per_seed]))
                point["classwise_ece"] = float(reduced["classwise_ece"])
                pooled[(condition, calibration)] = (per_seed, stats, point, reduced)
        for condition, calibration in pooled:
            per_seed, stats, point, reduced = pooled[(condition, calibration)]
            effect_stats = None
            if condition != "clean":
                clean_parts = pooled[("clean", calibration)][0]
                stacked_effects = []
                paired_rows = []
                for seed, clean_u, alt_u in zip(seeds, clean_parts, per_seed):
                    one = accumulate_effects(clean_u, alt_u, speaker_index, n_speakers)
                    alt_u["effect"] = {key: one.pop(f"u_{key}") for key in ("violations", "conf_gap", "cc_aw", "cw_ac")}
                    stacked_effects.append(one)
                    paired_rows.append(C["paired"](
                        _logit_frame(ids, clean_u, model, seed, "clean"),
                        _logit_frame(ids, alt_u, model, seed, condition),
                        1.0,
                    ))
                effect_stats = _stack_stats(stacked_effects)
                ours = effect_rates(effect_stats)
                agreed = _mean_dicts(paired_rows, EFFECT_METRICS)
                for key in EFFECT_METRICS:
                    if abs(float(ours[key]) - agreed[key]) > 1e-8:
                        raise ValueError(f"VRR statistics disagree with paired() for {model} {condition} {key}")
                    point[key] = agreed[key]
            blocks[(model, condition, calibration)] = Block(
                model=model, condition=condition, calibration=calibration, seeds=seeds,
                speaker_ids=speaker_ids, strata=np.asarray(strata), utterance_id=ids,
                speaker_index=speaker_index, folds=folds, stats=stats, effect_stats=effect_stats,
                seeds_utterances=per_seed, point=point,
                per_class_point={
                    "f1": np.asarray(reduced["per_class_f1"], dtype=np.float64),
                    "recall": np.asarray(reduced["per_class_recall"], dtype=np.float64),
                },
                partial=partial,
            )
        for per_seed, _stats, _point, _reduced in pooled.values():
            for parts in per_seed:
                parts.pop("effective_z", None)
    return Prepared(
        manifest=manifest, hashes=hashes, blocks=blocks, fold_scores=pd.DataFrame(fold_rows),
        partial=is_partial_manifest(manifest) or any(block.partial for block in blocks.values()),
        temperature_source=temperature_source, strata_column=strata_column, skipped_pairs=[],
    )


def _fold_differences(fold_scores: pd.DataFrame, model_a, model_b, condition, calibration, metric):
    if metric not in fold_scores.columns:
        return np.array([])
    sub = fold_scores[(fold_scores["condition"] == condition) & (fold_scores["calibration"] == calibration)]
    means = sub.groupby(["model", "fold"])[metric].mean()
    try:
        folds = sorted(set(means.loc[model_a].index) & set(means.loc[model_b].index))
    except KeyError:
        return np.array([])
    return np.array([float(means.loc[(model_b, fold)] - means.loc[(model_a, fold)]) for fold in folds], dtype=np.float64)


def _apply_holm(frame: pd.DataFrame, source: str, target: str) -> None:
    frame[target] = np.nan
    if frame.empty:
        return
    for _, index in frame.groupby(["family", "metric", "condition", "calibration"], sort=False).groups.items():
        ordered = list(index)
        frame.loc[ordered, target] = C["holm"](frame.loc[ordered, source].to_numpy(dtype=float))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


def _write_table(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    frame.loc[:, columns].to_csv(path, index=False)


def _assign_verdicts(frame: pd.DataFrame) -> None:
    if frame.empty:
        for column in ("bootstrap_verdict", "fold_t_verdict", "consensus_verdict"):
            frame[column] = pd.Series(dtype=object)
        return
    bootstrap, fold, consensus = [], [], []
    for row in frame.itertuples(index=False):
        bootstrap.append(bootstrap_component_verdict(
            row.ci90_low, row.ci90_high, row.ci95_low, row.ci95_high, row.metric
        ))
        fold.append(fold_component_verdict(row.fold_t_ci90_low, row.fold_t_ci90_high, row.holm_t_p, row.metric))
        consensus.append(consensus_verdict(
            row.ci90_low, row.ci90_high, row.ci95_low, row.ci95_high,
            row.fold_t_ci90_low, row.fold_t_ci90_high, row.holm_t_p, row.metric,
        ))
    frame["bootstrap_verdict"] = bootstrap
    frame["fold_t_verdict"] = fold
    frame["consensus_verdict"] = consensus


def run_strong_analysis(exports, out, temperatures=None, strata_column: str = "session",
                        n_boot: int = 10000, n_hboot: int = 2000, allow_partial: bool = False,
                        pairs_json=None, resampling: str = "unstratified") -> dict:
    if resampling not in ("unstratified", "stratified"):
        raise ValueError("resampling must be 'unstratified' or 'stratified'")
    if n_boot < 1 or n_hboot < 1:
        raise ValueError("Bootstrap counts must be positive")
    sensitivity_mode = "stratified" if resampling == "unstratified" else "unstratified"
    out = Path(out)
    if out.exists():
        raise ValueError(f"Refusing to overwrite existing output directory: {out}")
    exports = Path(exports).resolve()
    manifest = C["read_json"](exports / "export_manifest.json")
    if is_partial_manifest(manifest) and not allow_partial:
        raise ValueError("Partial export; pass --allow-partial for diagnostics only.")
    pairs = load_pairs(pairs_json)
    prepared = prepare_export(exports, temperatures, strata_column, allow_partial)
    blocks = prepared.blocks
    if not blocks:
        raise ValueError("No out-of-fold blocks were built")

    speaker_keys = sorted({block.speaker_ids for block in blocks.values()}, key=lambda ids: ids)
    boot_mult = {}
    sens_mult = {}
    hier_mult = {}
    for speaker_ids in speaker_keys:
        block = next(item for item in blocks.values() if item.speaker_ids == speaker_ids)
        boot_mult[speaker_ids] = speaker_multiplicities(block.strata, n_boot, BOOT_SEED, mode=resampling)
        sens_mult[speaker_ids] = speaker_multiplicities(block.strata, n_boot, BOOT_SEED, mode=sensitivity_mode)
        # Hierarchical resampling stays within stratum, as in protocol v1.
        hier_mult[speaker_ids] = speaker_multiplicities(block.strata, n_hboot, BOOT_SEED, mode="stratified")

    def _apply_multiplicities(mult_by_speakers):
        metric_boot, rate_boot = {}, {}
        for key, block in blocks.items():
            mult = mult_by_speakers[block.speaker_ids]
            metric_boot[key] = metrics_from_speaker_stats(block.stats, mult)
            if block.effect_stats is not None:
                rate_boot[key] = effect_rates(block.effect_stats, mult)
        return metric_boot, rate_boot

    boot, effect_boot = _apply_multiplicities(boot_mult)
    sens_boot, sens_effect = _apply_multiplicities(sens_mult)

    weight_cache = {}
    rng = np.random.default_rng(BOOT_SEED)
    for key in sorted({(block.speaker_ids, tuple(block.utterance_id.tolist())) for block in blocks.values()}):
        block = next(
            item for item in blocks.values()
            if item.speaker_ids == key[0] and tuple(item.utterance_id.tolist()) == key[1]
        )
        weight_cache[key] = hierarchical_weights(
            block.speaker_index, len(block.speaker_ids), hier_mult[block.speaker_ids], rng
        )

    hier = {}
    hier_effect = {}
    for key, block in blocks.items():
        weights = weight_cache[(block.speaker_ids, tuple(block.utterance_id.tolist()))]
        per_seed = [metrics_from_weights(utt, weights) for utt in block.seeds_utterances]
        hier[key] = {
            metric: np.mean([seed[metric] for seed in per_seed], axis=0)
            for metric in list(SCALAR_METRICS) + ["per_class_f1", "per_class_recall"]
        }
        if block.effect_stats is not None:
            rates = []
            count = weights.sum(axis=1)
            for utt in block.seeds_utterances:
                rates.append({
                    "vrr": np.divide(weights @ utt["effect"]["violations"], count, out=np.full(n_hboot, np.nan), where=count > 0),
                    "mean_clean_minus_altered_confidence": np.divide(
                        weights @ utt["effect"]["conf_gap"], count, out=np.full(n_hboot, np.nan), where=count > 0
                    ),
                    "clean_correct_altered_wrong_fraction": np.divide(
                        weights @ utt["effect"]["cc_aw"], count, out=np.full(n_hboot, np.nan), where=count > 0
                    ),
                    "clean_wrong_altered_correct_fraction": np.divide(
                        weights @ utt["effect"]["cw_ac"], count, out=np.full(n_hboot, np.nan), where=count > 0
                    ),
                })
            hier_effect[key] = {metric: np.mean([row[metric] for row in rates], axis=0) for metric in EFFECT_METRICS}

    point_rows, class_rows, effect_rows, hier_rows = [], [], [], []
    labels = C["LABELS"]
    for key, block in sorted(blocks.items()):
        intervals = {metric: percentile_intervals(boot[key][metric]) for metric in PAIR_METRICS}
        intervals["classwise_ece"] = percentile_intervals(boot[key]["classwise_ece"])
        for metric in SCALAR_METRICS:
            row = {
                "model": block.model, "condition": block.condition, "calibration": block.calibration,
                "metric": metric, "estimate": block.point[metric],
                "n_utterances": int(len(block.utterance_id)), "n_speakers": len(block.speaker_ids),
                "n_seeds": len(block.seeds), "n_folds": len(block.folds),
                "partial_folds": bool(block.partial),
                "ci_source": "speaker_bootstrap" if metric not in NONADDITIVE else "point_only",
            }
            if metric in NONADDITIVE:
                row.update({"ci95_low": np.nan, "ci95_high": np.nan, "ci90_low": np.nan, "ci90_high": np.nan})
            else:
                row.update(intervals[metric] if metric in intervals else percentile_intervals(boot[key][metric]))
            point_rows.append(row)
            h_int = percentile_intervals(hier[key][metric])
            hier_rows.append({
                "analysis": "metric", "model": block.model, "model_a": "", "model_b": "", "family": "",
                "condition": block.condition, "calibration": block.calibration, "metric": metric,
                "estimate": block.point[metric], **{k: h_int[k] for k in ("ci95_low", "ci95_high")},
            })
        for class_index, name in enumerate(labels):
            f1_samples = boot[key]["per_class_f1"][:, class_index]
            recall_samples = boot[key]["per_class_recall"][:, class_index]
            f1_ci = percentile_intervals(f1_samples)
            recall_ci = percentile_intervals(recall_samples)
            class_rows.append({
                "model": block.model, "condition": block.condition, "calibration": block.calibration,
                "class_index": class_index, "class_name": name,
                "f1": float(block.per_class_point["f1"][class_index]),
                "f1_ci95_low": f1_ci["ci95_low"], "f1_ci95_high": f1_ci["ci95_high"],
                "recall": float(block.per_class_point["recall"][class_index]),
                "recall_ci95_low": recall_ci["ci95_low"], "recall_ci95_high": recall_ci["ci95_high"],
                "partial_folds": bool(block.partial),
            })
        if block.condition != "clean":
            for metric in EFFECT_METRICS:
                ci = percentile_intervals(effect_boot[key][metric])
                effect_rows.append({
                    "model": block.model, "condition": block.condition, "calibration": block.calibration,
                    "metric": metric, "estimate": block.point[metric], **ci,
                    "partial_folds": bool(block.partial),
                })
                h_int = percentile_intervals(hier_effect[key][metric])
                hier_rows.append({
                    "analysis": "condition_effect", "model": block.model, "model_a": "", "model_b": "",
                    "family": "", "condition": block.condition, "calibration": block.calibration,
                    "metric": metric, "estimate": block.point[metric],
                    "ci95_low": h_int["ci95_low"], "ci95_high": h_int["ci95_high"],
                })

    present = {block.model for block in blocks.values()}
    skipped = []
    pair_rows = []
    sensitivity_rows = []
    mde_rows = []
    gee_cache = {}
    for model_a, model_b, family in pairs:
        if model_a not in present or model_b not in present:
            skipped.append({
                "model_a": model_a, "model_b": model_b, "family": family,
                "reason": "model missing from export; pair test skipped",
            })
            continue
        left = blocks[(model_a, "clean", "raw")]
        right = blocks[(model_b, "clean", "raw")]
        if left.speaker_ids != right.speaker_ids or not np.array_equal(left.utterance_id, right.utterance_id):
            skipped.append({
                "model_a": model_a, "model_b": model_b, "family": family,
                "reason": "utterance or speaker sets differ; pair test skipped",
            })
            continue
        if left.seeds != right.seeds:
            skipped.append({
                "model_a": model_a, "model_b": model_b, "family": family,
                "reason": "seed sets differ; pair test skipped",
            })
            continue
        shared = sorted(
            {cond for owned, cond, _cal in blocks if owned == model_a}
            & {cond for owned, cond, _cal in blocks if owned == model_b}
        )
        clean_diff = _fold_differences(prepared.fold_scores, model_a, model_b, "clean", "raw", "macro_f1")
        detected = minimum_detectable_effect(clean_diff)
        mde_rows.append({
            "model_a": model_a, "model_b": model_b, "metric": "macro_f1", "condition": "clean",
            "calibration": "raw", "n_folds": detected["n"], "sd_fold_diff": detected["sd"],
            "mde": detected["mde"], "note": detected["note"],
        })
        for condition in shared:
            for calibration in ("raw", "temperature"):
                gee_key = (model_a, model_b, condition, calibration)
                if gee_key not in gee_cache:
                    frames = []
                    for owned in (model_a, model_b):
                        block = blocks[(owned, condition, calibration)]
                        speakers = np.asarray(block.speaker_ids, dtype=object)
                        for utt in block.seeds_utterances:
                            frames.append(pd.DataFrame({
                                "speaker_id": speakers[block.speaker_index],
                                "model": owned,
                                "correct": utt["correct"],
                            }))
                    gee_cache[gee_key] = fit_gee(pd.concat(frames, ignore_index=True), model_a, model_b)
                gee = gee_cache[gee_key]
                metrics = list(PAIR_METRICS) + (["vrr"] if condition != "clean" else [])
                for metric in metrics:
                    if metric == "vrr":
                        samples = effect_boot[(model_b, condition, calibration)]["vrr"] - effect_boot[(model_a, condition, calibration)]["vrr"]
                        sens_samples = sens_effect[(model_b, condition, calibration)]["vrr"] - sens_effect[(model_a, condition, calibration)]["vrr"]
                        estimate = blocks[(model_b, condition, calibration)].point["vrr"] - blocks[(model_a, condition, calibration)].point["vrr"]
                    else:
                        samples = boot[(model_b, condition, calibration)][metric] - boot[(model_a, condition, calibration)][metric]
                        sens_samples = sens_boot[(model_b, condition, calibration)][metric] - sens_boot[(model_a, condition, calibration)][metric]
                        estimate = blocks[(model_b, condition, calibration)].point[metric] - blocks[(model_a, condition, calibration)].point[metric]
                    ci = percentile_intervals(samples)
                    tested = paired_t_test(_fold_differences(
                        prepared.fold_scores, model_a, model_b, condition, calibration, metric
                    ))
                    pair_rows.append({
                        "model_a": model_a, "model_b": model_b, "family": family,
                        "condition": condition, "calibration": calibration, "metric": metric,
                        "effect_b_minus_a": float(estimate), **ci,
                        "bootstrap_p": bootstrap_pvalue(samples),
                        "fold_t_effect": tested["effect"], "fold_t_df": tested["df"], "fold_t_p": tested["p"],
                        "fold_t_ci95_low": tested["ci95_low"], "fold_t_ci95_high": tested["ci95_high"],
                        "fold_t_ci90_low": tested["ci90_low"], "fold_t_ci90_high": tested["ci90_high"],
                        "fold_t_note": tested["note"], **gee,
                    })
                    if metric == "macro_f1":
                        sens_ci = percentile_intervals(sens_samples)
                        sensitivity_rows.append({
                            "model_a": model_a, "model_b": model_b, "family": family,
                            "condition": condition, "calibration": calibration, "metric": metric,
                            "effect_b_minus_a": float(estimate), **sens_ci,
                            "bootstrap_p": bootstrap_pvalue(sens_samples),
                            "resampling": sensitivity_mode,
                        })
                    if metric == "macro_f1":
                        h_samples = hier[(model_b, condition, calibration)][metric] - hier[(model_a, condition, calibration)][metric]
                        h_ci = percentile_intervals(h_samples)
                        hier_rows.append({
                            "analysis": "paired", "model": "", "model_a": model_a, "model_b": model_b,
                            "family": family, "condition": condition, "calibration": calibration,
                            "metric": metric, "estimate": float(estimate),
                            "ci95_low": h_ci["ci95_low"], "ci95_high": h_ci["ci95_high"],
                        })

    paired_df = pd.DataFrame(pair_rows)
    if not paired_df.empty:
        _apply_holm(paired_df, "bootstrap_p", "holm_bootstrap_p")
        _apply_holm(paired_df, "fold_t_p", "holm_t_p")
    _assign_verdicts(paired_df)

    clean_runs = prepared.fold_scores[
        (prepared.fold_scores["condition"] == "clean") & (prepared.fold_scores["calibration"] == "raw")
    ][["model", "seed", "fold", "macro_f1"]]
    variance = variance_components(clean_runs)

    slope_rows = []
    slope_models = []
    for model in sorted(present):
        if any((model, condition, "raw") not in boot for condition in LOSS_CONDITIONS):
            continue
        slope_models.append(model)
        point_y = np.array([blocks[(model, condition, "raw")].point["macro_f1"] for condition in LOSS_CONDITIONS])
        sample_y = np.column_stack([boot[(model, condition, "raw")]["macro_f1"] for condition in LOSS_CONDITIONS])
        samples = _slope(sample_y)
        slope_rows.append({
            "kind": "model", "model": model, "model_a": "", "model_b": "",
            "slope": float(_slope(point_y)), **percentile_intervals(samples),
        })
    for model_a, model_b, family in pairs:
        if model_a not in slope_models or model_b not in slope_models:
            continue
        if any(row["model_a"] == model_a and row["model_b"] == model_b and row.get("reason") for row in skipped):
            continue
        ya = np.column_stack([boot[(model_a, condition, "raw")]["macro_f1"] for condition in LOSS_CONDITIONS])
        yb = np.column_stack([boot[(model_b, condition, "raw")]["macro_f1"] for condition in LOSS_CONDITIONS])
        samples = _slope(yb) - _slope(ya)
        point = float(_slope(np.array([blocks[(model_b, c, "raw")].point["macro_f1"] for c in LOSS_CONDITIONS]))
                      - _slope(np.array([blocks[(model_a, c, "raw")].point["macro_f1"] for c in LOSS_CONDITIONS])))
        slope_rows.append({
            "kind": "difference", "model": "", "model_a": model_a, "model_b": model_b,
            "slope": point, **percentile_intervals(samples), "family": family,
        })

    limitations = list(LIMITATIONS)
    if resampling == "unstratified":
        limitations.append(
            "Primary speaker bootstrap resamples the full speaker set; within-session intervals are sensitivity only."
        )
    else:
        limitations.append(
            "Primary speaker bootstrap is within-session; unstratified intervals are sensitivity only."
        )
    if prepared.partial:
        limitations.append("Partial export: metrics use the available folds only and are flagged partial_folds.")
    if skipped:
        limitations.append("Pair tests that need a missing model, or models with different utterance sets, were skipped.")
    protocol = _repo_root() / "mtap_outputs" / "protocol" / "analysis_protocol_v1.md"
    amendment = _repo_root() / "mtap_outputs" / "protocol" / "analysis_protocol_v2_amendment.md"
    history = prepared.manifest.get("history_unknown_count")
    if history is None:
        history = sum(1 for run in prepared.manifest["runs"] if run.get("history_status") == "history_unknown")
    analysis_manifest = {
        "created": C["timestamp"](),
        "settings": {
            "exports": str(exports),
            "strata_column": strata_column,
            "temperatures": None if temperatures is None else str(temperatures),
            "temperature_source": prepared.temperature_source,
            "allow_partial": bool(allow_partial),
            "equivalence_margin": EQUIVALENCE_MARGIN,
            "ece_bins": N_BINS,
            "rng_seed": BOOT_SEED,
            "vrr_rule": "strict max(p_altered) > max(p_clean); exact ties not violations",
            "temperature_fit_split": "clean development only",
            "holm_grouping": "family, metric, condition, calibration",
            "dataset": prepared.manifest.get("dataset"),
            "resampling": resampling,
            "sensitivity_resampling": sensitivity_mode,
            "gee_covariance": "bias_reduced",
            "hierarchical_resampling": "stratified",
        },
        "B_speaker": int(n_boot),
        "B_hierarchical": int(n_hboot),
        "seeds": sorted({int(seed) for block in blocks.values() for seed in block.seeds}),
        "models": sorted(present),
        "partial": bool(prepared.partial),
        "input_file_sha256": prepared.hashes,
        "export_manifest_sha256": prepared.hashes["export_manifest.json"],
        "resampling": resampling,
        "sensitivity_resampling": sensitivity_mode,
        "protocol_sha256": C["sha256"](protocol),
        "protocol_amendment_sha256": C["sha256"](amendment),
        "history_unknown_count": int(history),
        "pairs_skipped": skipped,
        "limitations": limitations,
    }
    out.mkdir(parents=True, exist_ok=False)
    _write_table(pd.DataFrame(point_rows), out / "point_and_ci_metrics.csv", [
        "model", "condition", "calibration", "metric", "estimate", "ci95_low", "ci95_high",
        "ci90_low", "ci90_high", "ci_source", "n_utterances", "n_speakers", "n_seeds", "n_folds", "partial_folds",
    ])
    _write_table(paired_df, out / "paired_comparisons.csv", [
        "model_a", "model_b", "family", "condition", "calibration", "metric", "effect_b_minus_a",
        "ci95_low", "ci95_high", "ci90_low", "ci90_high", "bootstrap_p", "holm_bootstrap_p",
        "bootstrap_verdict", "fold_t_verdict", "consensus_verdict",
        "fold_t_effect", "fold_t_df", "fold_t_p", "holm_t_p",
        "fold_t_ci95_low", "fold_t_ci95_high", "fold_t_ci90_low", "fold_t_ci90_high",
        "fold_t_note", "gee_or", "gee_ci95_low", "gee_ci95_high", "gee_p", "gee_covariance", "gee_error",
        "gee_robust_or", "gee_robust_p",
    ])
    _write_table(pd.DataFrame(sensitivity_rows), out / "stratified_sensitivity.csv", [
        "model_a", "model_b", "family", "condition", "calibration", "metric", "resampling",
        "effect_b_minus_a", "ci95_low", "ci95_high", "ci90_low", "ci90_high", "bootstrap_p",
    ])
    _write_table(pd.DataFrame(class_rows), out / "per_class.csv", [
        "model", "condition", "calibration", "class_index", "class_name", "f1", "f1_ci95_low", "f1_ci95_high",
        "recall", "recall_ci95_low", "recall_ci95_high", "partial_folds",
    ])
    _write_table(pd.DataFrame(effect_rows), out / "condition_effects.csv", [
        "model", "condition", "calibration", "metric", "estimate", "ci95_low", "ci95_high",
        "ci90_low", "ci90_high", "partial_folds",
    ])
    _write_table(variance, out / "variance_components.csv", [
        "model", "n_folds", "n_seeds", "ss_fold", "ss_seed", "ss_residual", "var_fold", "var_seed",
        "var_residual", "negative_truncated", "note",
    ])
    _write_table(pd.DataFrame(mde_rows), out / "minimum_detectable_effects.csv", [
        "model_a", "model_b", "metric", "condition", "calibration", "n_folds", "sd_fold_diff", "mde", "note",
    ])
    _write_table(pd.DataFrame(slope_rows), out / "robustness_slopes.csv", [
        "kind", "model", "model_a", "model_b", "family", "slope", "ci95_low", "ci95_high", "ci90_low", "ci90_high",
    ])
    _write_table(pd.DataFrame(hier_rows), out / "hierarchical_sensitivity.csv", [
        "analysis", "model", "model_a", "model_b", "family", "condition", "calibration", "metric",
        "estimate", "ci95_low", "ci95_high",
    ])
    (out / "strong_analysis_manifest.json").write_text(
        json.dumps(_jsonable(analysis_manifest), indent=2) + "\n", encoding="utf-8"
    )
    clean = {
        model: {
            "estimate": blocks[(model, "clean", "raw")].point["macro_f1"],
            **percentile_intervals(boot[(model, "clean", "raw")]["macro_f1"]),
        }
        for model in sorted(present)
    }
    gee_failures = [] if paired_df.empty else sorted({
        row.gee_error for row in paired_df.itertuples() if isinstance(row.gee_error, str) and row.gee_error
    })
    return {
        "out": str(out),
        "n_runs": len(prepared.manifest["runs"]),
        "models": sorted(present),
        "partial": bool(prepared.partial),
        "history_unknown_count": int(history),
        "clean_raw_macro_f1": clean,
        "skipped_pairs": skipped,
        "gee_failures": gee_failures,
        "B_speaker": int(n_boot),
        "B_hierarchical": int(n_hboot),
        "resampling": resampling,
        "sensitivity_resampling": sensitivity_mode,
    }
