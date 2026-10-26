from __future__ import annotations

import torch
import torch.nn as nn

from .common import PredictionHeads, SequenceProjector, masked_mean


class UnimodalModel(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, num_classes: int,
        dropout: float = 0.2, temporal_stride: int = 1,
    ):
        super().__init__()
        self.temporal_stride = temporal_stride
        self.project = SequenceProjector(input_dim, hidden_dim, dropout)
        layer = nn.TransformerEncoderLayer(
            hidden_dim, 4, hidden_dim * 4, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.heads = PredictionHeads(hidden_dim, num_classes, dropout)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        values = values[:, :: self.temporal_stride]
        mask = mask[:, :: self.temporal_stride]
        encoded = self.encoder(self.project(values), src_key_padding_mask=~mask)
        return self.heads(masked_mean(encoded, mask))


class ConcatFusionModel(nn.Module):
    def __init__(
        self, audio_dim: int, video_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.audio = SequenceProjector(audio_dim, hidden_dim, dropout)
        self.video = SequenceProjector(video_dim, hidden_dim, dropout)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.heads = PredictionHeads(hidden_dim, num_classes, dropout)

    def forward(self, audio, audio_mask, video, video_mask) -> dict[str, torch.Tensor]:
        audio_pool = masked_mean(self.audio(audio), audio_mask)
        video_pool = masked_mean(self.video(video), video_mask)
        return self.heads(self.fuse(torch.cat([audio_pool, video_pool], dim=-1)))


class ConcatFusionModelMatched(nn.Module):
    """Capacity-matched concat baseline.

    Identical to :class:`ConcatFusionModel` except each modality is first
    passed through its own 2-layer pre-norm Transformer encoder (same depth,
    width, and head count as :class:`UnimodalModel`) before pooling, so the
    audio-only vs. concat comparison is not confounded by concat lacking the
    per-modality temporal-modeling capacity that the unimodal and Essence
    encoders have.
    """

    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.2,
        audio_stride: int = 4,
        fusion_width: int | None = None,
        encoder_layers: int = 2,
    ) -> None:
        super().__init__()
        if int(encoder_layers) < 1:
            raise ValueError("encoder_layers must be >= 1")
        width = hidden_dim if fusion_width is None else int(fusion_width)
        if width < 1:
            raise ValueError("fusion_width must be >= 1")
        self.audio_stride = audio_stride
        self.audio_project = SequenceProjector(audio_dim, hidden_dim, dropout)
        self.video_project = SequenceProjector(video_dim, hidden_dim, dropout)
        audio_layer = nn.TransformerEncoderLayer(
            hidden_dim, 4, hidden_dim * 4, dropout, batch_first=True, norm_first=True
        )
        video_layer = nn.TransformerEncoderLayer(
            hidden_dim, 4, hidden_dim * 4, dropout, batch_first=True, norm_first=True
        )
        self.audio_encoder = nn.TransformerEncoder(audio_layer, num_layers=int(encoder_layers))
        self.video_encoder = nn.TransformerEncoder(video_layer, num_layers=int(encoder_layers))
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, width), nn.GELU(), nn.Dropout(dropout)
        )
        self.heads = PredictionHeads(width, num_classes, dropout)

    def forward(
        self,
        audio,
        audio_mask,
        video,
        video_mask,
        *,
        disable_audio: bool = False,
        disable_video: bool = False,
    ) -> dict[str, torch.Tensor]:
        if disable_audio and disable_video:
            raise ValueError("cannot disable both audio and video branches")
        hidden_dim = self.fuse[0].in_features // 2
        if disable_audio:
            audio_pool = torch.zeros(audio.shape[0], hidden_dim, device=audio.device, dtype=audio.dtype)
        else:
            audio = audio[:, :: self.audio_stride]
            audio_mask = audio_mask[:, :: self.audio_stride]
            audio_encoded = self.audio_encoder(
                self.audio_project(audio), src_key_padding_mask=~audio_mask
            )
            audio_pool = masked_mean(audio_encoded, audio_mask)
        if disable_video:
            video_pool = torch.zeros(video.shape[0], hidden_dim, device=video.device, dtype=video.dtype)
        else:
            video_encoded = self.video_encoder(
                self.video_project(video), src_key_padding_mask=~video_mask
            )
            video_pool = masked_mean(video_encoded, video_mask)
        return self.heads(self.fuse(torch.cat([audio_pool, video_pool], dim=-1)))


class CrossAttentionBaseline(nn.Module):
    def __init__(
        self, audio_dim: int, video_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.audio_stride = 4
        self.audio = SequenceProjector(audio_dim, hidden_dim, dropout)
        self.video = SequenceProjector(video_dim, hidden_dim, dropout)
        self.audio_to_video = nn.MultiheadAttention(hidden_dim, 4, dropout, batch_first=True)
        self.video_to_audio = nn.MultiheadAttention(hidden_dim, 4, dropout, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU())
        self.heads = PredictionHeads(hidden_dim, num_classes, dropout)

    def forward(self, audio, audio_mask, video, video_mask) -> dict[str, torch.Tensor]:
        audio = audio[:, :: self.audio_stride]
        audio_mask = audio_mask[:, :: self.audio_stride]
        audio, video = self.audio(audio), self.video(video)
        audio_cross, av_attention = self.audio_to_video(
            audio, video, video, key_padding_mask=~video_mask, need_weights=True
        )
        video_cross, va_attention = self.video_to_audio(
            video, audio, audio, key_padding_mask=~audio_mask, need_weights=True
        )
        pooled = torch.cat(
            [masked_mean(audio + audio_cross, audio_mask), masked_mean(video + video_cross, video_mask)],
            dim=-1,
        )
        output = self.heads(self.fuse(pooled))
        output.update({"audio_to_video_attention": av_attention, "video_to_audio_attention": va_attention})
        return output
