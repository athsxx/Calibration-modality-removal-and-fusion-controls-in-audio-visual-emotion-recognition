from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import PredictionHeads, SequenceProjector, masked_mean


class EssenceTransferModel(nn.Module):
    """Directional cross-modal transfer with shared/private factors and reliability gating."""

    def __init__(
        self,
        audio_dim: int,
        video_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 4,
        num_heads: int = 4,
        fusion_layers: int = 2,
        dropout: float = 0.2,
        audio_stride: int = 4,
    ) -> None:
        super().__init__()
        self.audio_stride = audio_stride
        self.audio_project = SequenceProjector(audio_dim, hidden_dim, dropout)
        self.video_project = SequenceProjector(video_dim, hidden_dim, dropout)
        self.audio_shared = nn.Linear(hidden_dim, hidden_dim)
        self.video_shared = nn.Linear(hidden_dim, hidden_dim)
        self.audio_private = nn.Linear(hidden_dim, hidden_dim)
        self.video_private = nn.Linear(hidden_dim, hidden_dim)
        self.audio_receives_video = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout, batch_first=True
        )
        self.video_receives_audio = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout, batch_first=True
        )
        fusion_layer = nn.TransformerEncoderLayer(
            hidden_dim,
            num_heads,
            hidden_dim * 4,
            dropout,
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(fusion_layer, fusion_layers)
        self.reliability = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2)
        )
        self.heads = PredictionHeads(hidden_dim, num_classes, dropout)
        self.audio_teacher = nn.Linear(hidden_dim, hidden_dim)
        self.video_teacher = nn.Linear(hidden_dim, hidden_dim)

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
        audio = audio[:, :: self.audio_stride]
        audio_mask = audio_mask[:, :: self.audio_stride]
        hidden_dim = self.audio_shared.out_features
        batch_size = audio.shape[0]
        device, dtype = audio.device, audio.dtype

        if disable_audio:
            audio_pool = torch.zeros(batch_size, hidden_dim, device=device, dtype=dtype)
            video_base = self.video_project(video)
            video_shared = self.video_shared(video_base)
            video_private = self.video_private(video_base)
            video_cross = torch.zeros_like(video_shared)
            va_attention = None
            video_tokens = video_shared + video_private
            audio_tokens = torch.zeros(batch_size, audio.shape[1], hidden_dim, device=device, dtype=dtype)
            fused_tokens = video_tokens
            fused_mask = video_mask
            av_attention = None
        elif disable_video:
            video_pool = torch.zeros(batch_size, hidden_dim, device=device, dtype=dtype)
            audio_base = self.audio_project(audio)
            audio_shared = self.audio_shared(audio_base)
            audio_private = self.audio_private(audio_base)
            audio_cross = torch.zeros_like(audio_shared)
            av_attention = None
            audio_tokens = audio_shared + audio_private
            video_tokens = torch.zeros(batch_size, video.shape[1], hidden_dim, device=device, dtype=dtype)
            fused_tokens = audio_tokens
            fused_mask = audio_mask
            va_attention = None
        else:
            audio_base = self.audio_project(audio)
            video_base = self.video_project(video)
            audio_shared = self.audio_shared(audio_base)
            video_shared = self.video_shared(video_base)
            audio_private = self.audio_private(audio_base)
            video_private = self.video_private(video_base)
            audio_cross, av_attention = self.audio_receives_video(
                audio_shared,
                video_shared,
                video_shared,
                key_padding_mask=~video_mask,
                need_weights=True,
            )
            video_cross, va_attention = self.video_receives_audio(
                video_shared,
                audio_shared,
                audio_shared,
                key_padding_mask=~audio_mask,
                need_weights=True,
            )
            audio_tokens = audio_shared + audio_private + audio_cross
            video_tokens = video_shared + video_private + video_cross
            fused_tokens = torch.cat([audio_tokens, video_tokens], dim=1)
            fused_mask = torch.cat([audio_mask, video_mask], dim=1)

        fused_tokens = self.fusion(fused_tokens, src_key_padding_mask=~fused_mask)

        if disable_audio:
            video_pool = masked_mean(video_tokens, video_mask)
            fused_video = masked_mean(fused_tokens, video_mask)
            pooled = fused_video
            reliability = torch.tensor([[0.0, 1.0]], device=device, dtype=dtype).expand(batch_size, -1)
            audio_shared_pool = audio_pool
            video_shared_pool = masked_mean(video_shared, video_mask)
            audio_private_pool = audio_pool
            video_private_pool = masked_mean(video_private, video_mask)
        elif disable_video:
            audio_pool = masked_mean(audio_tokens, audio_mask)
            fused_audio = masked_mean(fused_tokens, audio_mask)
            pooled = fused_audio
            reliability = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype).expand(batch_size, -1)
            audio_shared_pool = masked_mean(audio_shared, audio_mask)
            video_shared_pool = video_pool
            audio_private_pool = masked_mean(audio_private, audio_mask)
            video_private_pool = video_pool
        else:
            audio_pool = masked_mean(audio_tokens, audio_mask)
            video_pool = masked_mean(video_tokens, video_mask)
            reliability = F.softmax(self.reliability(torch.cat([audio_pool, video_pool], dim=-1)), dim=-1)
            fused_audio = masked_mean(fused_tokens[:, : audio.shape[1]], audio_mask)
            fused_video = masked_mean(fused_tokens[:, audio.shape[1] :], video_mask)
            pooled = reliability[:, :1] * fused_audio + reliability[:, 1:] * fused_video
            audio_shared_pool = masked_mean(audio_shared, audio_mask)
            video_shared_pool = masked_mean(video_shared, video_mask)
            audio_private_pool = masked_mean(audio_private, audio_mask)
            video_private_pool = masked_mean(video_private, video_mask)

        output = self.heads(pooled)
        output.update(
            {
                "reliability": reliability,
                "audio_to_video_attention": av_attention,
                "video_to_audio_attention": va_attention,
                "audio_shared": audio_shared_pool,
                "video_shared": video_shared_pool,
                "audio_private": audio_private_pool,
                "video_private": video_private_pool,
                "audio_teaches_video": self.audio_teacher(audio_shared_pool),
                "video_teaches_audio": self.video_teacher(video_shared_pool),
            }
        )
        return output

    @torch.inference_mode()
    def counterfactual(self, batch: dict, drop: str) -> dict[str, torch.Tensor]:
        audio, video = batch["audio"], batch["video"]
        audio_mask, video_mask = batch["audio_mask"], batch["video_mask"]
        if drop == "audio":
            audio = torch.zeros_like(audio)
        elif drop == "video":
            video = torch.zeros_like(video)
        elif drop != "none":
            raise ValueError("drop must be one of: none, audio, video")
        return self(audio, audio_mask, video, video_mask)
