from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    recall_score,
)

from .data import EMOTION4_LABELS

# Default calibration settings (Guo et al. 2017 use M=15 equal-width bins).
CALIBRATION_N_BINS = 15
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
BOOTSTRAP_ALPHA = 0.05
PROB_EPS = 1e-12
TEMPERATURE_BOUNDS = (0.05, 50.0)


def _ccc(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_mean, target_mean = prediction.mean(), target.mean()
    covariance = np.mean((prediction - pred_mean) * (target - target_mean))
    return float(
        2 * covariance
        / (prediction.var() + target.var() + (pred_mean - target_mean) ** 2 + 1e-8)
    )


def calculate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    va_target: np.ndarray,
    va_prediction: np.ndarray,
    *,
    include_va: bool = True,
) -> dict:
    per_class_recall = recall_score(
        labels, predictions, labels=range(len(EMOTION4_LABELS)), average=None, zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "uar": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "per_class_recall": {
            label: float(value) for label, value in zip(EMOTION4_LABELS, per_class_recall)
        },
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=range(len(EMOTION4_LABELS))
        ).tolist(),
    }
    if include_va:
        metrics["valence_ccc"] = _ccc(va_prediction[:, 0], va_target[:, 0])
        metrics["activation_ccc"] = _ccc(va_prediction[:, 1], va_target[:, 1])
        metrics["va_mae"] = float(mean_absolute_error(va_target, va_prediction))
    else:
        metrics["valence_ccc"] = float("nan")
        metrics["activation_ccc"] = float("nan")
        metrics["va_mae"] = float("nan")
    return metrics


def _as_int_labels(labels: np.ndarray) -> np.ndarray:
    return np.asarray(labels, dtype=np.int64).reshape(-1)


def multiclass_nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class, $-\\frac{1}{n}\\sum_i \\log p_{i,y_i}$."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = _as_int_labels(labels)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (n, n_classes)")
    if len(labels) != probabilities.shape[0]:
        raise ValueError("labels and probabilities must have the same length")
    p_true = probabilities[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(p_true, PROB_EPS, 1.0))))


def multiclass_brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score: mean over samples of $\\sum_k (p_k - \\mathbf{1}[y=k])^2$."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = _as_int_labels(labels)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (n, n_classes)")
    n, n_classes = probabilities.shape
    if len(labels) != n:
        raise ValueError("labels and probabilities must have the same length")
    one_hot = np.zeros((n, n_classes), dtype=np.float64)
    one_hot[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def _ece_mce_from_bins(
    bin_ids: np.ndarray,
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int,
) -> tuple[float, float, list[dict]]:
    n = len(confidences)
    counts = np.bincount(bin_ids, minlength=n_bins).astype(np.float64)
    sum_conf = np.bincount(bin_ids, weights=confidences, minlength=n_bins)
    sum_acc = np.bincount(bin_ids, weights=correctness, minlength=n_bins)
    occupied = counts > 0
    acc = np.divide(sum_acc, counts, out=np.full(n_bins, np.nan), where=occupied)
    conf = np.divide(sum_conf, counts, out=np.full(n_bins, np.nan), where=occupied)
    gaps = np.abs(acc - conf)
    if n == 0 or not occupied.any():
        ece = float("nan")
        mce = float("nan")
    else:
        ece = float(np.sum((counts / n) * np.nan_to_num(gaps, nan=0.0)))
        mce = float(np.nanmax(gaps))
    rows = []
    for b in range(n_bins):
        rows.append(
            {
                "bin": b,
                "bin_confidence_mean": float(conf[b]) if occupied[b] else float("nan"),
                "bin_accuracy": float(acc[b]) if occupied[b] else float("nan"),
                "bin_count": int(counts[b]),
            }
        )
    return ece, mce, rows


def equal_width_ece_mce(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = CALIBRATION_N_BINS,
) -> tuple[float, float, list[dict]]:
    """Equal-width (fixed [0,1] grid) ECE and MCE on max-probability confidence."""
    confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
    correctness = np.asarray(correctness, dtype=np.float64).reshape(-1)
    n_bins = int(n_bins)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(confidences, bin_edges[1:-1], right=False)
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    ece, mce, rows = _ece_mce_from_bins(bin_ids, confidences, correctness, n_bins)
    for b, row in enumerate(rows):
        row["bin_lo"] = float(bin_edges[b])
        row["bin_hi"] = float(bin_edges[b + 1])
    return ece, mce, rows


def equal_mass_ece_mce(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = CALIBRATION_N_BINS,
) -> tuple[float, float, list[dict]]:
    """Equal-mass (adaptive / quantile) ECE and MCE on max-probability confidence.

    Samples are ranked by confidence and split into ``n_bins`` bins of as equal
    cardinality as possible. Empty bins (n < n_bins) are ignored in ECE/MCE.
    """
    confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
    correctness = np.asarray(correctness, dtype=np.float64).reshape(-1)
    n = len(confidences)
    n_bins = int(n_bins)
    if n == 0:
        return float("nan"), float("nan"), []
    order = np.argsort(confidences, kind="mergesort")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n)
    bin_ids = np.minimum((ranks * n_bins) // n, n_bins - 1)
    ece, mce, rows = _ece_mce_from_bins(bin_ids, confidences, correctness, n_bins)
    for b, row in enumerate(rows):
        mask = bin_ids == b
        if mask.any():
            row["bin_lo"] = float(confidences[mask].min())
            row["bin_hi"] = float(confidences[mask].max())
        else:
            row["bin_lo"] = float("nan")
            row["bin_hi"] = float("nan")
    return ece, mce, rows


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax(logits / T) with a numerically stable max-shift."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = np.asarray(logits, dtype=np.float64)
    z = logits / float(temperature)
    z = z - np.max(z, axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=-1, keepdims=True)


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    bounds: tuple[float, float] = TEMPERATURE_BOUNDS,
    n_grid: int = 48,
    n_refine: int = 30,
) -> float:
    """Fit a scalar softmax temperature by NLL minimization on *held-out* logits.

    This function must be called on a validation / development split, never on
    the evaluation (test) split that will be reported.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = _as_int_labels(labels)
    lo, hi = bounds
    if lo <= 0 or hi <= lo:
        raise ValueError("temperature bounds must satisfy 0 < lo < hi")

    def nll_at(temperature: float) -> float:
        return multiclass_nll(apply_temperature(logits, temperature), labels)

    grid = np.exp(np.linspace(np.log(lo), np.log(hi), n_grid))
    values = np.array([nll_at(float(t)) for t in grid])
    best = int(np.argmin(values))
    a = float(grid[max(0, best - 1)])
    b = float(grid[min(len(grid) - 1, best + 1)])
    phi = (1.0 + 5.0 ** 0.5) / 2.0
    for _ in range(n_refine):
        c = b - (b - a) / phi
        d = a + (b - a) / phi
        if nll_at(c) < nll_at(d):
            b = d
        else:
            a = c
    return float((a + b) / 2.0)


def classification_from_probabilities(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities.argmax(axis=-1)
    confidences = probabilities[np.arange(len(predictions)), predictions]
    return predictions, confidences


def calibration_report(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = CALIBRATION_N_BINS,
    include_bins: bool = False,
) -> dict:
    """Accuracy, NLL, Brier, equal-width ECE/MCE, and equal-mass ECE/MCE."""
    labels = _as_int_labels(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions, confidences = classification_from_probabilities(probabilities)
    correctness = (predictions == labels).astype(np.float64)
    n = len(labels)
    accuracy = float(correctness.mean()) if n else float("nan")
    ece_w, mce_w, bins_w = equal_width_ece_mce(confidences, correctness, n_bins=n_bins)
    ece_m, mce_m, bins_m = equal_mass_ece_mce(confidences, correctness, n_bins=n_bins)
    report = {
        "accuracy": accuracy,
        "nll": multiclass_nll(probabilities, labels) if n else float("nan"),
        "brier": multiclass_brier(probabilities, labels) if n else float("nan"),
        "ece_width": ece_w,
        "ece_mass": ece_m,
        "mce": mce_w,
        "mce_width": mce_w,
        "mce_mass": mce_m,
        "n_samples": int(n),
        "n_bins": int(n_bins),
    }
    if include_bins:
        report["bins"] = bins_w
        report["bins_mass"] = bins_m
    return report


def bootstrap_calibration_cis(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    n_bins: int = CALIBRATION_N_BINS,
    alpha: float = BOOTSTRAP_ALPHA,
    keys: tuple[str, ...] = ("accuracy", "nll", "brier", "ece_width", "ece_mass", "mce"),
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap CIs on pooled calibration metrics.

    Resamples utterances with replacement. Does not re-fit temperature.
    """
    labels = _as_int_labels(labels)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    n = len(labels)
    point = calibration_report(labels, probabilities, n_bins=n_bins)
    rng = np.random.default_rng(seed)
    collected = {key: np.empty(n_resamples, dtype=np.float64) for key in keys}
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sample = calibration_report(labels[idx], probabilities[idx], n_bins=n_bins)
        for key in keys:
            collected[key][i] = sample[key]
    lower_q = 100.0 * (alpha / 2.0)
    upper_q = 100.0 * (1.0 - alpha / 2.0)
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        out[key] = {
            "point": float(point[key]),
            "low": float(np.percentile(collected[key], lower_q)),
            "high": float(np.percentile(collected[key], upper_q)),
        }
    return out

