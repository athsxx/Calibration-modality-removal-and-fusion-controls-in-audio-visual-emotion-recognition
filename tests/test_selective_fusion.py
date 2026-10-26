"""Synthetic checks for protocol-v3 selective fusion. No test labels enter the gate."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mer_companion"))

import numpy as np

from iemocap_mer.selective_fusion import (
    LOGIT_WEIGHT_GRID,
    TAU_GRID,
    averaged_logits,
    choose_anchor,
    qmf_fusion,
    select_logit_weight,
    select_tau,
    selective_outputs,
    visual_evidence,
)
from merkit.analysis import classification, softmax


def test_tau_grid_and_tie_goes_to_larger_tau() -> None:
    assert TAU_GRID == tuple(round(0.30 + 0.05 * index, 2) for index in range(9))
    assert TAU_GRID[0] == 0.30 and TAU_GRID[-1] == 0.70
    labels = np.arange(4)
    probabilities = np.eye(4)
    tau, score = select_tau(np.full(4, 0.5), probabilities, probabilities, labels)
    assert tau == 0.70
    assert score == classification(labels, labels)["macro_f1"]


def test_smaller_tau_wins_when_it_is_strictly_better() -> None:
    labels = np.arange(4)
    anchor = np.eye(4)[[1, 2, 3, 0]]
    fused = np.eye(4)
    tau, score = select_tau(np.full(4, 0.32), anchor, fused, labels)
    assert tau == 0.30
    assert score == 1.0


def test_anchor_selection_and_audio_tie() -> None:
    assert choose_anchor({"audio_w2v": 0.40, "video_au": 0.55}) == "video_au"
    assert choose_anchor({"audio_w2v": 0.70, "video_au": 0.55}) == "audio_w2v"
    assert choose_anchor({"video_au": 0.50, "audio_w2v": 0.50}) == "audio_w2v"
    assert choose_anchor({"audio_w2v": 0.42}) == "audio_w2v"


def test_logit_average_weight_is_chosen_on_development_with_larger_tie() -> None:
    labels = np.arange(4)
    audio = np.eye(4) * 20.0
    video = np.eye(4)[[1, 2, 3, 0]] * 1.0
    weight, score = select_logit_weight(labels, audio, 1.0, video, 1.0)
    scores = {}
    for candidate in LOGIT_WEIGHT_GRID:
        fused = averaged_logits(audio, 1.0, video, 1.0, candidate)
        scores[candidate] = classification(labels, softmax(fused).argmax(1))["macro_f1"]
    expected = max(candidate for candidate, value in scores.items() if value == max(scores.values()))
    assert weight == expected == 1.0
    assert score == scores[weight]
    # A second call with the test labels shuffled must not be how the weight is chosen:
    # the function has no test split at all.
    again, _again_score = select_logit_weight(labels, audio, 1.0, video, 1.0)
    assert again == weight


def test_qmf_weights_sum_to_one() -> None:
    audio = np.array([[2.0, 0.0, -1.0, 0.5], [0.0, 3.0, 0.2, -0.4]])
    video = np.array([[0.0, 1.0, 0.0, -1.0], [-2.0, 0.5, 0.5, 0.0]])
    fused, weights = qmf_fusion(audio, 1.2, video, 0.8)
    assert weights.shape == (2, 2)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.all(weights > 0)
    manual_audio = audio / 1.2
    manual_video = video / 0.8
    assert np.allclose(fused, weights[:, 0:1] * manual_audio + weights[:, 1:2] * manual_video)


def test_zero_video_visual_evidence_is_zero() -> None:
    kept = np.array([8.0, 32.0, 1.0])
    assert np.all(visual_evidence("zero_video", kept) == 0.0)
    assert visual_evidence("branch_disabled_video", kept)[1] == 0.0
    assert visual_evidence("clean", np.array([16.0]))[0] == 0.5
    assert visual_evidence("video_token_loss_50", np.array([8.0]))[0] == 0.25


def test_shuffling_test_labels_does_not_change_selective_outputs() -> None:
    generator = np.random.default_rng(0)
    labels = np.tile(np.arange(4), 12)
    anchor = np.zeros((len(labels), 4))
    fused = np.zeros_like(anchor)
    anchor[np.arange(len(labels)), labels] = 4.0
    fused[np.arange(len(labels)), labels] = 4.0
    fused[:24] = 0.0
    fused[np.arange(24), (labels[:24] + 1) % 4] = 4.0
    anchor[24:] = 0.0
    anchor[np.arange(24, len(labels)), (labels[24:] + 1) % 4] = 5.0
    visual = generator.random(len(labels))
    development = {"y": labels, "anchor_z": anchor, "fused_z": fused, "visual": visual}
    test_labels = generator.integers(0, 4, size=16)
    test_anchor = generator.normal(size=(16, 4))
    test_fused = generator.normal(size=(16, 4))
    test = {
        "y": test_labels.copy(),
        "anchor_z": test_anchor,
        "fused_z": test_fused,
        "visual": generator.random(16),
    }
    first = selective_outputs(development, test)
    test["y"] = generator.permutation(test_labels)
    second = selective_outputs(development, test)
    assert first["tau"] == second["tau"]
    assert np.array_equal(first["logits"], second["logits"])
    assert np.allclose(first["probabilities"], softmax(first["logits"]))
