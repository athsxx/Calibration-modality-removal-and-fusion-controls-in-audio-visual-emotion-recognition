from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from torchvision import models, transforms
from tqdm import tqdm


def feature_path(
    root: str | Path,
    modality: str,
    utterance_id: str,
    *,
    feature_dir: str | None = None,
    session: int | None = None,
) -> Path:
    # IEMOCAP ids encode the session in characters [3:5] ("Ses01..."). Other
    # corpora pass session explicitly; the formatted directory stays Session{n}.
    if session is None:
        session = int(utterance_id[3:5])
    folder = feature_dir or modality
    return Path(root) / folder / f"Session{int(session)}" / f"{utterance_id}.pt"


def _row_session(row) -> int | None:
    value = getattr(row, "session", None)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return int(value)


class LogMelExtractor:
    output_dim: int

    def __init__(self, sample_rate: int = 16_000, n_mels: int = 80) -> None:
        self.sample_rate = sample_rate
        self.output_dim = n_mels
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=n_mels,
            f_min=20,
            f_max=sample_rate // 2,
            power=2.0,
        )

    @torch.inference_mode()
    def __call__(self, path: str | Path) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(str(path))
        waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        waveform = waveform / waveform.abs().max().clamp_min(1e-6)
        features = torch.log(self.transform(waveform).clamp_min(1e-6)).squeeze(0).transpose(0, 1)
        return (features - features.mean(dim=0)) / features.std(dim=0).clamp_min(1e-5)


class Wav2VecExtractor:
    """Compatible SSL encoder using torchaudio's Wav2Vec2 bundles."""

    def __init__(self, bundle_name: str = "WAV2VEC2_BASE", device: str = "cpu") -> None:
        bundle = getattr(torchaudio.pipelines, bundle_name)
        self.sample_rate = bundle.sample_rate
        self.model = bundle.get_model().eval().to(device)
        self.device = torch.device(device)
        self.output_dim = int(self.model.encoder.transformer.layers[0].attention.embed_dim)

    @torch.inference_mode()
    def __call__(self, path: str | Path) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(str(path))
        waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        features, _ = self.model.extract_features(waveform.to(self.device))
        return features[-1].squeeze(0).cpu()


class VideoExtractor:
    def __init__(
        self,
        fps: float = 4.0,
        max_frames: int = 32,
        pretrained: bool = False,
        device: str = "cpu",
        face_margin: float = 0.25,
        face_region: str = "speaker_half",
    ) -> None:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        network = models.resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(network.children())[:-1]).eval().to(device)
        self.output_dim = 512
        self.fps = fps
        self.max_frames = max_frames
        self.face_margin = face_margin
        if face_region not in {"speaker_half", "full_frame"}:
            raise ValueError(f"Unknown face_region: {face_region}")
        self.face_region = face_region
        self.device = torch.device(device)
        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        cascade_root = cv2.data.haarcascades
        self.face_detector = cv2.CascadeClassifier(
            cascade_root + "haarcascade_frontalface_default.xml"
        )
        self.profile_detector = cv2.CascadeClassifier(
            cascade_root + "haarcascade_profileface.xml"
        )

    def _times(self, start: float, end: float) -> np.ndarray:
        count = max(1, min(self.max_frames, int(np.ceil((end - start) * self.fps))))
        return np.linspace(start, end, count, endpoint=False) + (end - start) / (2 * count)

    def _speaker_half(self, frame: np.ndarray, gender: str) -> np.ndarray:
        if self.face_region == "full_frame":
            return frame
        width = frame.shape[1]
        return frame[:, : width // 2] if gender == "F" else frame[:, width // 2 :]

    def _detect_faces(self, gray: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        for image in (enhanced, gray):
            faces = self.face_detector.detectMultiScale(
                image, scaleFactor=1.08, minNeighbors=5, minSize=(48, 48)
            )
            if len(faces):
                return faces
            faces = self.profile_detector.detectMultiScale(
                image, scaleFactor=1.08, minNeighbors=5, minSize=(48, 48)
            )
            if len(faces):
                return faces
            faces = self.face_detector.detectMultiScale(
                image, scaleFactor=1.05, minNeighbors=3, minSize=(32, 32)
            )
            if len(faces):
                return faces
        return np.empty((0, 4), dtype=np.int32)

    def _face_box(self, image: np.ndarray) -> tuple[int, int, int, int] | None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._detect_faces(gray)
        if len(faces):
            x, y, width, height = max(faces, key=lambda item: item[2] * item[3])
            margin = int(self.face_margin * max(width, height))
            x0, y0 = max(0, x - margin), max(0, y - margin)
            x1, y1 = min(image.shape[1], x + width + margin), min(
                image.shape[0], y + height + margin
            )
            return x0, y0, x1, y1
        return None

    def _face_crop(
        self, image: np.ndarray, box: tuple[int, int, int, int] | None = None
    ) -> tuple[np.ndarray, bool]:
        box = box or self._face_box(image)
        if box is not None:
            x0, y0, x1, y1 = box
            return image[y0:y1, x0:x1], True
        height, width = image.shape[:2]
        side = min(height, width)
        x0, y0 = (width - side) // 2, (height - side) // 2
        return image[y0 : y0 + side, x0 : x0 + side], False

    @torch.inference_mode()
    def __call__(
        self, video_path: str | Path, start: float, end: float, speaker_gender: str
    ) -> tuple[torch.Tensor, dict]:
        capture = cv2.VideoCapture(str(video_path))
        tensors: list[torch.Tensor] = []
        detections = 0
        for time_seconds in self._times(start, end):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(time_seconds * 1000.0))
            ok, frame = capture.read()
            if not ok:
                continue
            crop, detected = self._face_crop(self._speaker_half(frame, speaker_gender))
            detections += int(detected)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.preprocess(rgb))
        capture.release()
        if not tensors:
            raise RuntimeError(f"Could not decode frames from {video_path} at {start:.3f}-{end:.3f}")
        batch = torch.stack(tensors).to(self.device)
        features = self.encoder(batch).flatten(1).cpu()
        quality = {
            "frames": len(tensors),
            "face_detections": detections,
            "face_detection_rate": detections / len(tensors),
        }
        return features, quality

    @torch.inference_mode()
    def encode_frames(self, frames: list[torch.Tensor]) -> torch.Tensor:
        return self.encoder(torch.stack(frames).to(self.device)).flatten(1).cpu()


@dataclass
class ExtractionSummary:
    requested: int = 0
    extracted: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[dict] | None = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []


def extract_features(
    frame: pd.DataFrame,
    output_root: str | Path,
    modality: str,
    extractor,
    overwrite: bool = False,
    *,
    feature_dir: str | None = None,
) -> dict:
    output_root = Path(output_root)
    folder = feature_dir or modality
    summary = ExtractionSummary(requested=len(frame))
    quality_rows: list[dict] = []
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=f"extract-{folder}"):
        destination = feature_path(
            output_root, modality, row.utterance_id, feature_dir=folder, session=_row_session(row)
        )
        if destination.exists() and not overwrite:
            summary.skipped += 1
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            if modality == "audio":
                tensor = extractor(row.audio_path)
                quality = {}
            elif modality == "video":
                tensor, quality = extractor(
                    row.video_path, row.start_seconds, row.end_seconds, row.speaker_gender
                )
            else:
                raise ValueError(f"Unknown modality: {modality}")
            if tensor.ndim != 2 or tensor.shape[0] == 0:
                raise ValueError(f"Invalid feature shape: {tuple(tensor.shape)}")
            torch.save(tensor.to(torch.float32), destination)
            summary.extracted += 1
            quality_rows.append({"utterance_id": row.utterance_id, **quality})
        except Exception as exc:  # keep a complete audit instead of aborting a long extraction
            summary.failed += 1
            summary.errors.append({"utterance_id": row.utterance_id, "error": str(exc)})

    report = {
        "requested": summary.requested,
        "extracted": summary.extracted,
        "skipped": summary.skipped,
        "failed": summary.failed,
        "errors": summary.errors,
    }
    report_path = output_root / f"{folder}_extraction_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if quality_rows:
        pd.DataFrame(quality_rows).to_csv(output_root / f"{folder}_quality.csv", index=False)
    return report


def extract_video_features_batched(
    frame: pd.DataFrame,
    output_root: str | Path,
    extractor: VideoExtractor,
    overwrite: bool = False,
    encoder_batch_size: int = 64,
    video_dir: str = "video",
) -> dict:
    """Extract video features with one video open per dialogue and batched inference."""
    output_root = Path(output_root)
    report = {"requested": int(len(frame)), "extracted": 0, "skipped": 0, "failed": 0, "errors": []}
    quality_rows: list[dict] = []
    progress = tqdm(total=len(frame), desc="extract-video-batched")

    for _, group in frame.groupby("dialogue_id", sort=False):
        pending_rows = []
        for row in group.itertuples(index=False):
            destination = feature_path(
                output_root, "video", row.utterance_id, feature_dir=video_dir, session=_row_session(row)
            )
            if destination.exists() and not overwrite:
                report["skipped"] += 1
                progress.update(1)
            else:
                pending_rows.append(row)
        if not pending_rows:
            continue

        capture = cv2.VideoCapture(str(pending_rows[0].video_path))
        encoded: dict[str, list[torch.Tensor]] = {row.utterance_id: [] for row in pending_rows}
        detections = {row.utterance_id: 0 for row in pending_rows}
        attempted = {row.utterance_id: 0 for row in pending_rows}
        batch_frames: list[torch.Tensor] = []
        batch_ids: list[str] = []

        def flush() -> None:
            if not batch_frames:
                return
            values = extractor.encode_frames(batch_frames)
            for utterance_id, value in zip(batch_ids, values):
                encoded[utterance_id].append(value)
            batch_frames.clear()
            batch_ids.clear()

        requests = sorted(
            (
                float(time_seconds),
                row.utterance_id,
                row.speaker_gender,
            )
            for row in pending_rows
            for time_seconds in extractor._times(row.start_seconds, row.end_seconds)
        )
        face_boxes: dict[str, tuple[int, int, int, int] | None] = {
            row.utterance_id: None for row in pending_rows
        }
        video_fps = float(capture.get(cv2.CAP_PROP_FPS) or 29.97)
        current_frame = -1
        decode_ok = True
        for time_seconds, utterance_id, speaker_gender in requests:
            attempted[utterance_id] += 1
            target_frame = max(current_frame, int(round(time_seconds * video_fps)))
            while current_frame < target_frame:
                decode_ok = capture.grab()
                current_frame += 1
                if not decode_ok:
                    break
            if not decode_ok:
                break
            ok, image = capture.retrieve()
            if not ok:
                continue
            speaker_view = extractor._speaker_half(image, speaker_gender)
            if face_boxes[utterance_id] is None:
                face_boxes[utterance_id] = extractor._face_box(speaker_view)
            crop, detected = extractor._face_crop(
                speaker_view, face_boxes[utterance_id]
            )
            detections[utterance_id] += int(detected)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            batch_frames.append(extractor.preprocess(rgb))
            batch_ids.append(utterance_id)
            if len(batch_frames) >= encoder_batch_size:
                flush()
        flush()
        capture.release()

        for row in pending_rows:
            values = encoded[row.utterance_id]
            if not values:
                report["failed"] += 1
                report["errors"].append(
                    {"utterance_id": row.utterance_id, "error": "No video frames decoded"}
                )
            else:
                destination = feature_path(
                    output_root, "video", row.utterance_id, feature_dir=video_dir, session=_row_session(row)
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                torch.save(torch.stack(values).to(torch.float32), destination)
                report["extracted"] += 1
                quality_rows.append(
                    {
                        "utterance_id": row.utterance_id,
                        "frames": len(values),
                        "attempted_frames": attempted[row.utterance_id],
                        "face_tracked_frames": detections[row.utterance_id],
                        "face_track_rate": detections[row.utterance_id]
                        / max(1, len(values)),
                    }
                )
            progress.update(1)
    progress.close()
    (output_root / "video_extraction_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    quality_path = output_root / f"{video_dir}_quality.csv"
    pd.DataFrame(quality_rows).to_csv(quality_path, index=False)
    return report


def verify_feature_cache(
    frame: pd.DataFrame, output_root: str | Path, modality: str, *, feature_dir: str | None = None
) -> dict:
    folder = feature_dir or modality
    missing, invalid, dimensions = [], [], {}
    for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=f"verify-{folder}"):
        utterance_id = row.utterance_id
        path = feature_path(
            output_root, modality, utterance_id, feature_dir=folder, session=_row_session(row)
        )
        if not path.exists():
            missing.append(utterance_id)
            continue
        try:
            value = torch.load(path, map_location="cpu", weights_only=True)
            if value.ndim != 2 or value.shape[0] < 1 or not torch.isfinite(value).all():
                invalid.append({"utterance_id": utterance_id, "shape": list(value.shape)})
            else:
                dimensions[str(int(value.shape[1]))] = dimensions.get(str(int(value.shape[1])), 0) + 1
        except Exception as exc:
            invalid.append({"utterance_id": utterance_id, "error": str(exc)})
    report = {
        "modality": modality,
        "expected": int(len(frame)),
        "valid": int(len(frame) - len(missing) - len(invalid)),
        "missing_count": len(missing),
        "invalid_count": len(invalid),
        "feature_dimensions": dimensions,
        "missing": missing,
        "invalid": invalid,
    }
    report_path = Path(output_root) / f"{modality}_verification_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
