import torch

from iemocap_mer.losses import multitask_loss
from iemocap_mer.models import (
    ConcatFusionModel,
    ConcatFusionModelMatched,
    CrossAttentionBaseline,
    EssenceTransferModel,
    UnimodalModel,
)


def test_all_model_variants_and_losses() -> None:
    batch, audio_time, video_time = 3, 12, 5
    audio = torch.randn(batch, audio_time, 8)
    video = torch.randn(batch, video_time, 10)
    audio_mask = torch.ones(batch, audio_time, dtype=torch.bool)
    video_mask = torch.ones(batch, video_time, dtype=torch.bool)
    labels = torch.tensor([0, 1, 2])
    va = torch.randn(batch, 2).tanh()
    weights = {key: 1.0 for key in (
        "emotion", "va", "alignment", "orthogonality",
        "audio_to_video_distillation", "video_to_audio_distillation"
    )}
    models = [
        (UnimodalModel(8, 16, 4), True),
        (ConcatFusionModel(8, 10, 16, 4), False),
        (ConcatFusionModelMatched(8, 10, 16, 4), False),
        (CrossAttentionBaseline(8, 10, 16, 4), False),
        (EssenceTransferModel(8, 10, 16, 4, 4, 1), False),
    ]
    for model, unimodal in models:
        output = model(audio, audio_mask) if unimodal else model(
            audio, audio_mask, video, video_mask
        )
        losses = multitask_loss(output, labels, va, weights)
        assert output["logits"].shape == (batch, 4)
        assert torch.isfinite(losses["total"])
        losses["total"].backward()

    matched = ConcatFusionModelMatched(8, 10, 16, 4)
    essence = EssenceTransferModel(8, 10, 16, 4, 4, 1)
    for model in (matched, essence):
        for disable_audio, disable_video in ((True, False), (False, True)):
            output = model(
                audio,
                audio_mask,
                video,
                video_mask,
                disable_audio=disable_audio,
                disable_video=disable_video,
            )
            assert output["logits"].shape == (batch, 4)
            assert torch.isfinite(output["logits"]).all()
