import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mer_companion"))
from merkit.analysis import classification, paired, score

from iemocap_mer.robust_stats import (
    BOOT_SEED,
    _speaker_axis,
    aurc,
    bootstrap_component_verdict,
    confusion_matrix,
    consensus_verdict,
    dersimonian_laird,
    fit_gee,
    fold_component_verdict,
    macro_f1_contrast,
    metrics_from_confusion,
    metrics_unit_weight,
    speaker_effect_rates,
    speaker_multiplicities,
    utterance_parts,
    weighted_confusion,
)

PY = sys.executable
SCORE_KEYS = ("macro_f1", "uar", "accuracy", "nll", "brier", "ece_width", "mce_width")


def _logits(ids, labels, probabilities, condition="clean", model="audio_w2v"):
    probabilities = np.asarray(probabilities, dtype=float)
    logits = np.log(probabilities)
    frame = pd.DataFrame({
        "model": model, "seed": 0, "fold": 1, "utterance_id": ids,
        "condition": condition, "split": "test", "label": labels,
    })
    for k in range(4):
        frame[f"p_{k}"] = probabilities[:, k]
        frame[f"z_{k}"] = logits[:, k]
    return frame, logits


def test_weighted_confusion_matches_resampled_dataset():
    labels_a = np.array([0, 1, 1, 2])
    pred_a = np.array([0, 1, 0, 2])
    labels_b = np.array([2, 3, 2, 3, 1])
    pred_b = np.array([2, 2, 3, 3, 1])
    combined = weighted_confusion(
        np.stack([confusion_matrix(labels_a, pred_a), confusion_matrix(labels_b, pred_b)]),
        np.array([2.0, 1.0]),
    )
    labels = np.concatenate([labels_a, labels_a, labels_b])
    predicted = np.concatenate([pred_a, pred_a, pred_b])
    expected = classification(labels, predicted)["macro_f1"]
    assert abs(metrics_from_confusion(combined)["macro_f1"] - expected) < 1e-12


def test_unit_multiplicities_match_companion_score():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(48, 4))
    labels = rng.integers(0, 4, size=48)
    got = metrics_unit_weight(labels, logits)
    expected = score(labels, logits)
    for key in SCORE_KEYS:
        assert abs(got[key] - expected[key]) < 1e-12
    assert abs(got["ece_mass"] - expected["ece_mass"]) < 1e-12
    parts = utterance_parts(labels, logits)
    direct = aurc(parts["conf"], parts["correct"])
    weighted = aurc(parts["conf"], parts["correct"], np.ones(len(labels)))
    assert abs(direct - weighted) < 1e-8


def test_identical_models_are_equivalent_and_draws_are_shared():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 4, size=24)
    predicted = labels.copy()
    speaker = np.repeat(np.arange(4), 6)
    strata = np.array([0, 0, 1, 1])
    result = macro_f1_contrast(labels, predicted, predicted, speaker, strata, n_boot=40, seed=BOOT_SEED)
    assert result["effect"] == 0.0
    assert result["verdict"] == "equivalent"
    assert np.all(result["samples"] == 0.0)
    again = speaker_multiplicities(strata, 40, BOOT_SEED)
    assert np.array_equal(result["mult"], again)
    other = np.where(predicted == 0, 1, predicted)
    second = macro_f1_contrast(labels, predicted, other, speaker, strata, n_boot=40, seed=BOOT_SEED)
    assert np.array_equal(result["mult"], second["mult"])


def test_planted_difference_excludes_zero():
    labels = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    speaker = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    strata = np.array([10, 20])
    worse = np.zeros(len(labels), dtype=int)
    result = macro_f1_contrast(labels, worse, labels, speaker, strata, n_boot=40, seed=BOOT_SEED)
    assert result["verdict"] == "different"
    assert result["ci95_low"] > 0.0 or result["ci95_high"] < 0.0
    assert result["effect"] > 0.2


def test_strata_size_is_preserved():
    strata = np.array([1, 1, 2, 2, 3, 3])
    draws = speaker_multiplicities(strata, 30, BOOT_SEED, mode="stratified")
    repeat = speaker_multiplicities(strata, 30, BOOT_SEED, mode="stratified")
    assert np.array_equal(draws, repeat)
    assert not np.array_equal(draws, speaker_multiplicities(strata, 30, BOOT_SEED + 1, mode="stratified"))
    for level in (1, 2, 3):
        assert np.all(draws[:, strata == level].sum(axis=1) == (strata == level).sum())


def test_unstratified_draws_are_shared_and_keep_speaker_count():
    strata = np.array([1, 1, 2, 2, 3, 3])
    first = speaker_multiplicities(strata, 25, BOOT_SEED, mode="unstratified")
    second = speaker_multiplicities(strata, 25, BOOT_SEED, mode="unstratified")
    assert np.array_equal(first, second)
    assert np.all(first.sum(axis=1) == len(strata))
    labels = np.arange(12) % 4
    speaker = np.repeat(np.arange(6), 2)
    predicted = labels.copy()
    other = (labels + 1) % 4
    left = macro_f1_contrast(labels, predicted, other, speaker, strata, n_boot=25, seed=BOOT_SEED, mode="unstratified")
    right = macro_f1_contrast(labels, other, predicted, speaker, strata, n_boot=25, seed=BOOT_SEED, mode="unstratified")
    assert np.array_equal(left["mult"], right["mult"])
    assert np.array_equal(left["mult"], first)
    assert np.all(left["mult"].sum(axis=1) == len(strata))


def test_consensus_verdict_truth_table():
    cases = [
        # Both procedures exclude no effect and the intervals are outside the margin.
        dict(metric="macro_f1", ci90=(0.04, 0.09), ci95=(0.03, 0.10), fold90=(0.02, 0.11), holm=0.01,
             bootstrap="different", fold="different", consensus="different"),
        # Bootstrap excludes 0; the Holm fold test does not.
        dict(metric="macro_f1", ci90=(0.01, 0.06), ci95=(0.005, 0.07), fold90=(-0.04, 0.08), holm=0.20,
             bootstrap="different", fold="inconclusive", consensus="inconclusive"),
        # Both 90% intervals lie inside the margin and the 95% interval includes 0.
        dict(metric="macro_f1", ci90=(-0.01, 0.01), ci95=(-0.015, 0.015), fold90=(-0.012, 0.012), holm=0.40,
             bootstrap="equivalent", fold="equivalent", consensus="equivalent"),
        # Only the bootstrap 90% interval is inside the margin.
        dict(metric="uar", ci90=(-0.01, 0.01), ci95=(-0.012, 0.012), fold90=(0.0, 0.05), holm=0.30,
             bootstrap="equivalent", fold="inconclusive", consensus="inconclusive"),
        # Not a margin metric: agreement on a nonzero effect is 'different', never 'equivalent'.
        dict(metric="nll", ci90=(-0.01, 0.01), ci95=(0.005, 0.015), fold90=(-0.01, 0.01), holm=0.01,
             bootstrap="different", fold="different", consensus="different"),
        # Both the equivalence and the difference criteria hold; the margin decides the claim.
        dict(metric="accuracy", ci90=(0.006, 0.014), ci95=(0.005, 0.015), fold90=(0.001, 0.015), holm=0.01,
             bootstrap="equivalent", fold="equivalent", consensus="equivalent"),
    ]
    assert len(cases) >= 5
    for case in cases:
        assert bootstrap_component_verdict(*case["ci90"], *case["ci95"], case["metric"]) == case["bootstrap"]
        assert fold_component_verdict(*case["fold90"], case["holm"], case["metric"]) == case["fold"]
        assert consensus_verdict(*case["ci90"], *case["ci95"], *case["fold90"], case["holm"], case["metric"]) == case["consensus"]


def test_vrr_counts_match_companion_paired():
    clean, z_clean = _logits(
        ["a", "b", "c", "d"], [0, 1, 2, 3],
        [[0.7, 0.1, 0.1, 0.1], [0.2, 0.5, 0.2, 0.1], [0.1, 0.1, 0.6, 0.2], [0.4, 0.2, 0.2, 0.2]],
    )
    altered, z_altered = _logits(
        ["a", "b", "c", "d"], [0, 1, 2, 3],
        [[0.5, 0.3, 0.1, 0.1], [0.2, 0.5, 0.2, 0.1], [0.05, 0.05, 0.1, 0.8], [0.6, 0.1, 0.1, 0.2]],
        condition="zero_audio",
    )
    reference = paired(clean, altered.iloc[::-1])
    got = speaker_effect_rates(
        clean["label"].to_numpy(), z_clean, z_altered, np.array([0, 0, 1, 1])
    )
    parts_clean = utterance_parts(clean["label"].to_numpy(), z_clean)
    parts_altered = utterance_parts(clean["label"].to_numpy(), z_altered)
    assert got["violations"] == int(np.sum(parts_altered["conf"] > parts_clean["conf"]))
    assert got["n"] == reference["n"]
    for key in (
        "vrr",
        "mean_clean_minus_altered_confidence",
        "clean_correct_altered_wrong_fraction",
        "clean_wrong_altered_correct_fraction",
    ):
        assert abs(got[key] - reference[key]) < 1e-12


def test_dersimonian_laird_matches_hand_calculation():
    homogeneous = dersimonian_laird([0.1, 0.2, 0.3], [0.1, 0.1, 0.1])
    assert abs(homogeneous["q"] - 2.0) < 1e-12
    assert homogeneous["tau2"] == 0.0
    assert homogeneous["i2"] == 0.0
    assert abs(homogeneous["pooled"] - 0.2) < 1e-12
    assert abs(homogeneous["se"] - (1.0 / 300.0) ** 0.5) < 1e-12
    heterogeneous = dersimonian_laird([0.0, 1.0], [0.1, 0.1])
    assert abs(heterogeneous["q"] - 50.0) < 1e-12
    assert abs(heterogeneous["tau2"] - 0.49) < 1e-12
    assert abs(heterogeneous["i2"] - 0.98) < 1e-12
    assert abs(heterogeneous["pooled"] - 0.5) < 1e-12
    assert abs(heterogeneous["se"] - 0.5) < 1e-12
    assert heterogeneous["ci95_low"] < heterogeneous["pooled"] < heterogeneous["ci95_high"]


def test_gee_runs_on_clustered_binary_outcome():
    rng = np.random.default_rng(2)
    rows = []
    for speaker in range(6):
        for model, rate in (("anchor", 0.25), ("fusion", 0.8)):
            draws = rng.random(40)
            rows.extend(
                {"speaker_id": f"s{speaker}", "model": model, "correct": int(draw < rate)}
                for draw in draws
            )
    fitted = fit_gee(pd.DataFrame(rows), "anchor", "fusion")
    assert fitted["gee_error"] == ""
    assert fitted["gee_covariance"] == "bias_reduced"
    assert fitted["gee_or"] > 1.0
    assert 0.0 < fitted["gee_p"] < 0.01
    assert fitted["gee_ci95_low"] > 1.0
    assert fitted["gee_robust_or"] > 1.0
    assert 0.0 < fitted["gee_robust_p"] < fitted["gee_p"]


def test_speaker_axis_accepts_integer_speaker_ids():
    meta = pd.DataFrame({
        "utterance_id": ["a", "b", "c"],
        "speaker_id": [1, 1, 2],
        "session": [2, 2, 1],
    }).set_index("utterance_id", drop=False)
    speaker_ids, strata, speaker_index = _speaker_axis(np.array(["c", "a", "b"]), meta, "session")
    assert speaker_ids == ("2", "1")
    assert list(strata) == [1, 2]
    assert list(speaker_index) == [0, 1, 1]


def test_cli_refuses_partial_and_overwrite(tmp_path):
    exports = tmp_path / "export"
    exports.mkdir()
    (exports / "export_manifest.json").write_text(json.dumps({
        "complete": True, "selected_count": 1, "expected_principal_count": 75,
        "runs": [], "history_unknown_count": 0, "dataset": "iemocap", "settings": {},
    }), encoding="utf-8")
    out = tmp_path / "out"
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    partial = subprocess.run(
        [PY, "scripts/mtap_strong_analysis.py", "--exports", str(exports), "--out", str(out)],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert partial.returncode == 2
    assert "partial" in partial.stderr.lower()
    assert not out.exists()
    out.mkdir()
    overwrite = subprocess.run(
        [PY, "scripts/mtap_strong_analysis.py", "--exports", str(exports), "--out", str(out), "--allow-partial"],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    assert overwrite.returncode == 2
    assert "overwrite" in overwrite.stderr.lower()
