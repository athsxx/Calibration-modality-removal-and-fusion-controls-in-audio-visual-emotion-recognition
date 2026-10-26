"""Step 9 controls are opt-in. Defaults keep the published architectures and losses."""

from pathlib import Path

import pytest
import torch

from iemocap_mer.config import load_config
from iemocap_mer.losses import multitask_loss
from iemocap_mer.models import ConcatFusionModelMatched, EssenceTransferModel
from iemocap_mer.training import active_loss_weights, build_model

REPO = Path(__file__).resolve().parents[1]
DISTILLATION = ("audio_to_video_distillation", "video_to_audio_distillation")
TOLERANCE = 0.02
BUDGET_LAYERS = 1
BUDGET_WIDTH = 2032


def _count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def test_default_concat_matched_matches_explicit_published_width() -> None:
    torch.manual_seed(0)
    left = ConcatFusionModelMatched(768, 20, 256, 4, dropout=0.0)
    right = ConcatFusionModelMatched(
        768, 20, 256, 4, dropout=0.0, fusion_width=256, encoder_layers=2
    )
    assert [tuple(value.shape) for value in left.state_dict().values()] == [
        tuple(value.shape) for value in right.state_dict().values()
    ]
    right.load_state_dict(left.state_dict())
    audio = torch.randn(2, 16, 768)
    video = torch.randn(2, 8, 20)
    audio_mask = torch.ones(2, 16, dtype=torch.bool)
    video_mask = torch.ones(2, 8, dtype=torch.bool)
    with torch.no_grad():
        first = left(audio, audio_mask, video, video_mask)["logits"]
        second = right(audio, audio_mask, video, video_mask)["logits"]
    assert torch.equal(first, second)
    assert len(left.audio_encoder.layers) == 2
    assert left.fuse[0].out_features == 256


CHECKPOINT = REPO / "artifacts/checkpoints/concat_matched_fold_1_w2v_au_seed0.pt"


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="model weights are not distributed")
def test_existing_concat_checkpoint_loads_into_the_default_model() -> None:
    path = CHECKPOINT
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert "fusion_width" not in checkpoint["model_config"]
    assert "encoder_layers" not in checkpoint["model_config"]
    model = build_model(
        checkpoint["model_name"],
        checkpoint["audio_dim"],
        checkpoint["video_dim"],
        checkpoint["model_config"],
    )
    model.load_state_dict(checkpoint["state_dict"])


def test_distillation_flag_defaults_on_and_zero_weight_drops_both_terms() -> None:
    loaded = load_config(REPO / "configs/w2v_video_au.yaml")
    training = loaded["training"]
    assert training["directional_distillation"] is True
    stage2 = active_loss_weights(training, "essence", epoch=20)
    assert stage2["audio_to_video_distillation"] == 0.02
    assert stage2["video_to_audio_distillation"] == 0.02
    assert stage2["alignment"] == 0.02
    off = dict(training)
    off["directional_distillation"] = False
    silenced = active_loss_weights(off, "essence", epoch=20)
    assert silenced["audio_to_video_distillation"] == 0.0
    assert silenced["video_to_audio_distillation"] == 0.0
    assert silenced["emotion"] == stage2["emotion"]
    assert silenced["alignment"] == stage2["alignment"]
    stage1 = active_loss_weights(off, "essence", epoch=1)
    assert stage1["audio_to_video_distillation"] == 0.0
    assert stage1["video_to_audio_distillation"] == 0.0

    torch.manual_seed(1)
    model = EssenceTransferModel(8, 10, 16, 4, num_heads=4, fusion_layers=1, dropout=0.0)
    audio = torch.randn(3, 12, 8)
    video = torch.randn(3, 5, 10)
    output = model(
        audio,
        torch.ones(3, 12, dtype=torch.bool),
        video,
        torch.ones(3, 5, dtype=torch.bool),
    )
    labels = torch.tensor([0, 1, 2])
    va = torch.zeros(3, 2)
    on_terms = multitask_loss(output, labels, va, stage2)
    off_terms = multitask_loss(output, labels, va, silenced)
    dropped = sum(stage2[name] * on_terms[name] for name in DISTILLATION)
    assert torch.allclose(on_terms["total"], off_terms["total"] + dropped)
    off_terms["total"].backward(retain_graph=True)
    assert torch.count_nonzero(model.audio_teacher.weight.grad) == 0
    assert torch.count_nonzero(model.video_teacher.weight.grad) == 0
    model.zero_grad(set_to_none=True)
    on_terms["total"].backward()
    assert torch.count_nonzero(model.audio_teacher.weight.grad) > 0
    assert torch.count_nonzero(model.video_teacher.weight.grad) > 0


def test_nodist_yaml_changes_only_the_flag() -> None:
    pairs = (
        ("configs/w2v_video_au.yaml", "configs/iemocap_w2v_video_au_nodist.yaml"),
        ("configs/ravdess_w2v_video_au.yaml", "configs/ravdess_w2v_video_au_nodist.yaml"),
        ("configs/cremad_w2v_video_au.yaml", "configs/cremad_w2v_video_au_nodist.yaml"),
    )
    for source_name, nodist_name in pairs:
        source = (REPO / source_name).read_text(encoding="utf-8").splitlines()
        altered = (REPO / nodist_name).read_text(encoding="utf-8").splitlines()
        extra = [line for line in altered if line not in source]
        missing = [line for line in source if line not in altered]
        assert missing == []
        assert extra == ["  directional_distillation: false"]
        assert load_config(REPO / nodist_name)["training"]["directional_distillation"] is False


def test_budget_yaml_changes_only_depth_and_width_and_matches_essence() -> None:
    pairs = (
        ("configs/w2v_video_au.yaml", "configs/iemocap_w2v_video_au_budget.yaml"),
        ("configs/ravdess_w2v_video_au.yaml", "configs/ravdess_w2v_video_au_budget.yaml"),
        ("configs/cremad_w2v_video_au.yaml", "configs/cremad_w2v_video_au_budget.yaml"),
    )
    for source_name, budget_name in pairs:
        source = (REPO / source_name).read_text(encoding="utf-8").splitlines()
        altered = (REPO / budget_name).read_text(encoding="utf-8").splitlines()
        extra = [line for line in altered if line not in source]
        missing = [line for line in source if line not in altered]
        assert missing == []
        assert extra == ["  encoder_layers: 1", "  fusion_width: 2032"]
        loaded = load_config(REPO / budget_name)["model"]
        assert loaded["encoder_layers"] == BUDGET_LAYERS
        assert loaded["fusion_width"] == BUDGET_WIDTH
        assert load_config(REPO / source_name)["model"].get("fusion_width") is None

    for video_dim in (20, 512):
        essence = _count(EssenceTransferModel(
            768, video_dim, 256, 4, num_heads=4, fusion_layers=2, dropout=0.2
        ))
        published = _count(ConcatFusionModelMatched(768, video_dim, 256, 4, dropout=0.2))
        assert published > essence
        head_only = _count(ConcatFusionModelMatched(
            768, video_dim, 256, 4, dropout=0.2, fusion_width=1
        ))
        assert abs(head_only - essence) / essence > TOLERANCE
        matched = _count(ConcatFusionModelMatched(
            768, video_dim, 256, 4, dropout=0.2,
            fusion_width=BUDGET_WIDTH, encoder_layers=BUDGET_LAYERS,
        ))
        assert abs(matched - essence) / essence <= TOLERANCE
        neighbour = _count(ConcatFusionModelMatched(
            768, video_dim, 256, 4, dropout=0.2,
            fusion_width=BUDGET_WIDTH - 1, encoder_layers=BUDGET_LAYERS,
        ))
        assert abs(matched - essence) < abs(neighbour - essence)
