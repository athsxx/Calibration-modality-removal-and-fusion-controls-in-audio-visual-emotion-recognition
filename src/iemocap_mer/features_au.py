from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .features import VideoExtractor, _row_session, feature_path

AU_COLUMNS = [
    "AU01",
    "AU02",
    "AU04",
    "AU05",
    "AU06",
    "AU07",
    "AU09",
    "AU10",
    "AU11",
    "AU12",
    "AU14",
    "AU15",
    "AU17",
    "AU20",
    "AU23",
    "AU24",
    "AU25",
    "AU26",
    "AU28",
    "AU43",
]


def _resize_rgb(frame: np.ndarray, size: int = 224) -> np.ndarray:
    if frame.shape[0] == size and frame.shape[1] == size:
        return frame
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)


class VideoAUExtractor:
    """OpenFace-style AU features via py-feat on v2 face crops."""

    def __init__(
        self,
        fps: float = 6.0,
        max_frames: int = 32,
        face_margin: float = 0.35,
        device: str = "cpu",
        batch_size: int = 16,
        face_region: str = "speaker_half",
    ) -> None:
        from feat import Detectorv1

        self.cropper = VideoExtractor(
            fps=fps,
            max_frames=max_frames,
            pretrained=False,
            device="cpu",
            face_margin=face_margin,
            face_region=face_region,
        )
        self.detector = Detectorv1(
            device=device,
            emotion_model=None,
            identity_model=None,
            gaze_model=None,
        )
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.output_dim = len(AU_COLUMNS)
        self.fps = fps
        self.max_frames = max_frames

    def _times(self, start: float, end: float) -> np.ndarray:
        return self.cropper._times(start, end)

    def _speaker_half(self, frame: np.ndarray, gender: str) -> np.ndarray:
        return self.cropper._speaker_half(frame, gender)

    def _face_box(self, image: np.ndarray) -> tuple[int, int, int, int] | None:
        return self.cropper._face_box(image)

    def _face_crop(
        self, image: np.ndarray, box: tuple[int, int, int, int] | None = None
    ) -> tuple[np.ndarray, bool]:
        return self.cropper._face_crop(image, box)

    @torch.inference_mode()
    def encode_rgb_frames(self, rgb_frames: list[np.ndarray]) -> torch.Tensor:
        if not rgb_frames:
            raise ValueError("No frames to encode")
        features: list[torch.Tensor] = []
        for start in range(0, len(rgb_frames), self.batch_size):
            batch = [_resize_rgb(frame) for frame in rgb_frames[start : start + self.batch_size]]
            tensors = [
                torch.from_numpy(frame).permute(2, 0, 1).float() for frame in batch
            ]
            stacked = torch.stack(tensors)
            result = self.detector.detect(
                stacked,
                data_type="tensor",
                batch_size=len(batch),
                progress_bar=False,
            )
            values = result[AU_COLUMNS].to_numpy(dtype=np.float32)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(torch.from_numpy(values))
        return torch.cat(features, dim=0)

    @torch.inference_mode()
    def __call__(
        self, video_path: str | Path, start: float, end: float, speaker_gender: str
    ) -> tuple[torch.Tensor, dict]:
        capture = cv2.VideoCapture(str(video_path))
        rgb_frames: list[np.ndarray] = []
        detections = 0
        for time_seconds in self._times(start, end):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(time_seconds * 1000.0))
            ok, frame = capture.read()
            if not ok:
                continue
            crop, detected = self._face_crop(self._speaker_half(frame, speaker_gender))
            detections += int(detected)
            rgb_frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        capture.release()
        if not rgb_frames:
            raise RuntimeError(f"Could not decode frames from {video_path} at {start:.3f}-{end:.3f}")
        features = self.encode_rgb_frames(rgb_frames)
        quality = {
            "frames": len(rgb_frames),
            "face_detections": detections,
            "face_detection_rate": detections / len(rgb_frames),
        }
        return features, quality


def extract_video_au_features_batched(
    frame: pd.DataFrame,
    output_root: str | Path,
    extractor: VideoAUExtractor,
    overwrite: bool = False,
    video_dir: str = "video_au",
) -> dict:
    """Extract AU features with one video open per dialogue and batched py-feat inference."""
    output_root = Path(output_root)
    report = {"requested": int(len(frame)), "extracted": 0, "skipped": 0, "failed": 0, "errors": []}
    quality_rows: list[dict] = []
    progress = tqdm(total=len(frame), desc="extract-video-au-batched")

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
        batch_frames: list[np.ndarray] = []
        batch_ids: list[str] = []

        def flush() -> None:
            if not batch_frames:
                return
            values = extractor.encode_rgb_frames(batch_frames)
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
            crop, detected = extractor._face_crop(speaker_view, face_boxes[utterance_id])
            detections[utterance_id] += int(detected)
            batch_frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            batch_ids.append(utterance_id)
            if len(batch_frames) >= extractor.batch_size:
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
