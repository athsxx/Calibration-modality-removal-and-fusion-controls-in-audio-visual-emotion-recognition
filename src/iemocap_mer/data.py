from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .features import feature_path


EMOTION4_LABELS = ("angry", "happy_excited", "neutral", "sad")
LABEL_TO_INDEX = {label: index for index, label in enumerate(EMOTION4_LABELS)}


class MultimodalFeatureDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        utterance_ids: list[str],
        feature_root: str | Path,
        load_audio: bool = True,
        load_video: bool = True,
        audio_max_tokens: int = 400,
        video_max_tokens: int = 32,
        audio_dir: str = "audio",
        video_dir: str = "video",
        has_va: bool = True,
    ) -> None:
        lookup = frame.set_index("utterance_id", drop=False)
        self.rows = lookup.loc[utterance_ids].reset_index(drop=True)
        self.feature_root = Path(feature_root)
        self.load_audio = load_audio
        self.load_video = load_video
        self.audio_max_tokens = audio_max_tokens
        self.video_max_tokens = video_max_tokens
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.has_va = bool(has_va)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows.iloc[index]
        if self.has_va:
            va = torch.tensor(
                [
                    (min(5.0, max(1.0, float(row.valence))) - 3.0) / 2.0,
                    (min(5.0, max(1.0, float(row.activation))) - 3.0) / 2.0,
                ],
                dtype=torch.float32,
            )
        else:
            va = torch.zeros(2, dtype=torch.float32)
        item = {
            "utterance_id": row.utterance_id,
            "label": LABEL_TO_INDEX[row.emotion4],
            "va": va,
            "has_va": self.has_va,
        }
        for modality, load in (("audio", self.load_audio), ("video", self.load_video)):
            if modality == "video":
                feature_dir = self.video_dir
            elif modality == "audio":
                feature_dir = self.audio_dir
            else:
                feature_dir = None
            session = int(row.session) if "session" in row.index and pd.notna(row.session) else None
            path = feature_path(
                self.feature_root, modality, row.utterance_id, feature_dir=feature_dir, session=session
            )
            if load and path.exists():
                tensor = torch.load(path, map_location="cpu", weights_only=True)
                item[modality] = torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)
                limit = self.audio_max_tokens if modality == "audio" else self.video_max_tokens
                if item[modality].shape[0] > limit:
                    indices = torch.linspace(
                        0, item[modality].shape[0] - 1, limit
                    ).round().long()
                    item[modality] = item[modality][indices]
            elif load:
                raise FileNotFoundError(path)
            else:
                item[modality] = torch.zeros(1, 1)
        return item


def _pad(items: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([value.shape[0] for value in items], dtype=torch.long)
    padded = pad_sequence(items, batch_first=True)
    mask = torch.arange(padded.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    return padded, mask


def multimodal_collate(batch: list[dict]) -> dict:
    audio, audio_mask = _pad([item["audio"] for item in batch])
    video, video_mask = _pad([item["video"] for item in batch])
    return {
        "utterance_id": [item["utterance_id"] for item in batch],
        "audio": audio,
        "audio_mask": audio_mask,
        "video": video,
        "video_mask": video_mask,
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "va": torch.stack([item["va"] for item in batch]),
        "has_va": all(bool(item.get("has_va", True)) for item in batch),
    }
