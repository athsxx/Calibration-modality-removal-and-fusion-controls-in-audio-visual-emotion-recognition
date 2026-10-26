from pathlib import Path

import torch

from iemocap_mer.config import load_config
from iemocap_mer.losses import (
    CML_FIXED_WEIGHT,
    cml_regularization_loss,
    combine_task_and_cml,
    faithful_cml_objective,
    faithful_training_loss,
    multitask_loss,
    predictive_confidence,
)
from iemocap_mer.models import ConcatFusionModelMatched
from iemocap_mer.training import (
    batch_with_zeroed_modality,
    cml_penalty_for_batch,
    faithful_cml_for_batch,
    forward_model,
)

REPO = Path(__file__).resolve().parents[1]


def test_cml_off_returns_the_same_task_loss() -> None:
    logits = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    labels = torch.tensor([0, 1])
    va = torch.zeros(2, 2)
    task = multitask_loss( {"logits": logits, "va": va}, labels, va, {"emotion": 1.0, "va": 0.2})["total"]
    penalty = torch.tensor(0.4)
    assert combine_task_and_cml(task, 0.0, penalty) is task
    assert combine_task_and_cml(task, 0.0) is task


def test_cml_penalty_is_nonnegative_and_zero_when_removed_is_not_higher() -> None:
    full = torch.tensor([0.80, 0.40, 0.90])
    tied_or_lower = torch.tensor([0.80, 0.40, 0.10])
    higher = torch.tensor([0.91, 0.40, 0.90])
    quiet = cml_regularization_loss(full, tied_or_lower, tied_or_lower)
    assert float(quiet) == 0.0
    active = cml_regularization_loss(full, tied_or_lower, higher)
    assert float(active) > 0.0
    expected = torch.relu(higher - full).mean()
    assert torch.allclose(active, expected)
    task = torch.tensor(1.0)
    weighted = combine_task_and_cml(task, CML_FIXED_WEIGHT, active)
    assert float(weighted) >= float(task)
    assert torch.allclose(weighted, task + CML_FIXED_WEIGHT * active)
    assert torch.equal(combine_task_and_cml(task, 1.0, quiet), task)


def test_zero_fill_matches_export_removal_and_penalty_backprops() -> None:
    torch.manual_seed(0)
    model = ConcatFusionModelMatched(8, 10, 16, 4, dropout=0.0)
    audio = torch.randn(3, 12, 8)
    video = torch.randn(3, 5, 10)
    batch = {
        "audio": audio,
        "audio_mask": torch.ones(3, 12, dtype=torch.bool),
        "video": video,
        "video_mask": torch.ones(3, 5, dtype=torch.bool),
    }
    full = forward_model(model, "concat_matched", batch)
    for modality in ("audio", "video"):
        altered = batch_with_zeroed_modality(batch, modality)
        assert altered[modality].shape == batch[modality].shape
        assert torch.count_nonzero(altered[modality]) == 0
        assert altered[f"{modality}_mask"] is batch[f"{modality}_mask"]
        assert torch.equal(batch[modality], audio if modality == "audio" else video)
    penalty = cml_penalty_for_batch(model, "concat_matched", batch, full["logits"])
    assert float(penalty.detach()) >= 0.0
    assert torch.isfinite(penalty.detach())
    penalty.backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_predictive_confidence_is_max_softmax() -> None:
    logits = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    conf = predictive_confidence(logits)
    assert torch.allclose(conf[0], torch.tensor(0.5))
    assert torch.allclose(conf[1], torch.softmax(logits[1], dim=-1).amax())


def test_existing_configs_leave_cml_off_and_cml_yaml_changes_only_the_weight() -> None:
    pairs = (
        ("configs/w2v_video_au.yaml", "configs/iemocap_w2v_video_au_cml.yaml"),
        ("configs/ravdess_w2v_video_au.yaml", "configs/ravdess_w2v_video_au_cml.yaml"),
        ("configs/cremad_w2v_video_au.yaml", "configs/cremad_w2v_video_au_cml.yaml"),
    )
    for source_name, cml_name in pairs:
        source = (REPO / source_name).read_text(encoding="utf-8").splitlines()
        altered = (REPO / cml_name).read_text(encoding="utf-8").splitlines()
        assert "cml_weight" not in "\n".join(source)
        extra = [line for line in altered if line not in source]
        missing = [line for line in source if line not in altered]
        assert missing == []
        assert extra == ["  cml_weight: 1.0"]
        loaded = load_config(REPO / source_name)
        assert loaded["training"]["cml_weight"] == 0.0
        assert loaded["training"]["cml_variant"] == "penalty"
        assert "cml_variant" not in "\n".join(source)
        assert "cml_variant" not in "\n".join(altered)
        assert load_config(REPO / cml_name)["training"]["cml_weight"] == CML_FIXED_WEIGHT == 1.0
        assert load_config(REPO / cml_name)["training"]["cml_variant"] == "penalty"


def test_faithful_yaml_adds_only_the_variant_flag() -> None:
    pairs = (
        ("configs/iemocap_w2v_video_au_cml.yaml", "configs/iemocap_w2v_video_au_cmlfaithful.yaml"),
        ("configs/ravdess_w2v_video_au_cml.yaml", "configs/ravdess_w2v_video_au_cmlfaithful.yaml"),
        ("configs/cremad_w2v_video_au_cml.yaml", "configs/cremad_w2v_video_au_cmlfaithful.yaml"),
    )
    for penalty_name, faithful_name in pairs:
        penalty = (REPO / penalty_name).read_text(encoding="utf-8").splitlines()
        faithful = (REPO / faithful_name).read_text(encoding="utf-8").splitlines()
        extra = [line for line in faithful if line not in penalty]
        missing = [line for line in penalty if line not in faithful]
        assert missing == []
        assert extra == ["  cml_variant: faithful"]
        loaded = load_config(REPO / faithful_name)
        assert loaded["training"]["cml_weight"] == CML_FIXED_WEIGHT == 1.0
        assert loaded["training"]["cml_variant"] == "faithful"


def test_faithful_objective_is_algorithm_1_for_two_modalities() -> None:
    ce_full = torch.tensor(1.2)
    ce_partial = torch.tensor(0.8)
    conf_full = torch.tensor([0.70, 0.40])
    conf_partial = torch.tensor([0.90, 0.40])
    hinge = torch.relu(conf_partial - conf_full).mean()
    loss = faithful_cml_objective(
        ce_full, [ce_partial], conf_full, [conf_partial], cml_weight=1.0
    )
    assert torch.allclose(loss, (ce_full + ce_partial) / 2 + hinge)
    unweighted = faithful_cml_objective(
        ce_full, [ce_partial], conf_full, [conf_partial], cml_weight=0.0
    )
    assert torch.allclose(unweighted, (ce_full + ce_partial) / 2)
    # A second removal in the chain is added, not averaged into the hinge.
    conf_next = torch.tensor([0.95, 0.10])
    chained = faithful_cml_objective(
        ce_full, [ce_partial, ce_partial], conf_full, [conf_partial, conf_next], cml_weight=1.0
    )
    chain_hinge = hinge + torch.relu(conf_next - conf_partial).mean()
    assert torch.allclose(chained, (ce_full + ce_partial + ce_partial) / 3 + chain_hinge)


def test_faithful_keeps_nonclassification_loss_on_the_full_input() -> None:
    ce_full = torch.tensor(1.0)
    ce_partial = torch.tensor(0.4)
    va = torch.tensor(0.5)
    task = ce_full + 0.2 * va
    conf_full = torch.tensor([0.20])
    conf_partial = torch.tensor([0.50])
    term = faithful_cml_objective(
        ce_full, [ce_partial], conf_full, [conf_partial], cml_weight=1.0
    )
    total = faithful_training_loss(task, ce_full, 1.0, term)
    hinge = torch.relu(conf_partial - conf_full).mean()
    expected = 0.2 * va + 0.5 * (ce_full + ce_partial) + hinge
    assert torch.allclose(total, expected)


def test_faithful_hinge_gradient_is_not_detached() -> None:
    logits_full = torch.tensor([[2.0, 0.0, 0.0, 0.0]], requires_grad=True)
    logits_partial = torch.tensor([[0.0, 3.0, 0.0, 0.0]], requires_grad=True)
    conf_full = predictive_confidence(logits_full)
    conf_partial = predictive_confidence(logits_partial)
    assert float(conf_partial.detach()) > float(conf_full.detach())
    hinge_loss = faithful_cml_objective(
        logits_full.new_zeros(()),
        [logits_partial.new_zeros(())],
        conf_full,
        [conf_partial],
        cml_weight=1.0,
    )
    hinge_loss.backward()
    assert logits_full.grad is not None and float(logits_full.grad.abs().sum()) > 0.0
    assert logits_partial.grad is not None and float(logits_partial.grad.abs().sum()) > 0.0


def test_faithful_batch_zero_fills_only_the_sampled_modality() -> None:
    torch.manual_seed(0)
    model = ConcatFusionModelMatched(8, 10, 16, 4, dropout=0.0)
    audio = torch.randn(3, 12, 8)
    video = torch.randn(3, 5, 10)
    labels = torch.tensor([0, 1, 2])
    batch = {
        "audio": audio,
        "audio_mask": torch.ones(3, 12, dtype=torch.bool),
        "video": video,
        "video_mask": torch.ones(3, 5, dtype=torch.bool),
    }
    full = forward_model(model, "concat_matched", batch)
    loss = faithful_cml_for_batch(
        model,
        "concat_matched",
        batch,
        full["logits"],
        labels,
        modality="audio",
        emotion_weight=1.0,
        cml_weight=1.0,
    )
    altered = batch_with_zeroed_modality(batch, "audio")
    assert torch.count_nonzero(altered["audio"]) == 0
    assert torch.equal(altered["video"], video)
    assert altered["audio_mask"] is batch["audio_mask"]
    assert torch.equal(batch["audio"], audio)
    partial_logits = forward_model(model, "concat_matched", altered)["logits"]
    expected = faithful_cml_objective(
        torch.nn.functional.cross_entropy(full["logits"], labels),
        [torch.nn.functional.cross_entropy(partial_logits, labels)],
        predictive_confidence(full["logits"]),
        [predictive_confidence(partial_logits)],
        cml_weight=1.0,
    )
    assert torch.allclose(loss, expected)
    assert float(loss.detach()) >= 0.0
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
