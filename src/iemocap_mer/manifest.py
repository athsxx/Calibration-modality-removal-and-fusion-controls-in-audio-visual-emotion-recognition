from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


SUMMARY_RE = re.compile(
    r"^\[(?P<start>[\d.]+)\s*-\s*(?P<end>[\d.]+)\]\s+"
    r"(?P<utterance>\S+)\s+(?P<emotion>\S+)\s+"
    r"\[(?P<valence>[\d.]+),\s*(?P<activation>[\d.]+),\s*(?P<dominance>[\d.]+)\]"
)
TRANSCRIPT_RE = re.compile(
    r"^(?P<utterance>\S+)\s+\[[^]]+\]:\s*(?P<transcript>.*)$"
)
EMOTION_NAMES = {
    "ang": "angry",
    "dis": "disgust",
    "exc": "excited",
    "fea": "fear",
    "fru": "frustrated",
    "hap": "happy",
    "neu": "neutral",
    "oth": "other",
    "sad": "sad",
    "sur": "surprised",
    "xxx": "no_majority",
}
EMOTION4 = {
    "ang": "angry",
    "hap": "happy_excited",
    "exc": "happy_excited",
    "neu": "neutral",
    "sad": "sad",
}


@dataclass(frozen=True)
class UtteranceRecord:
    utterance_id: str
    dialogue_id: str
    session: int
    speaker_id: str
    speaker_gender: str
    marker_actor_gender: str
    scenario_type: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    emotion_code: str
    emotion: str
    emotion4: str
    valence: float
    activation: float
    dominance: float
    distress_proxy: bool
    transcript: str
    audio_path: str
    video_path: str
    evaluation_path: str
    audio_exists: bool
    video_exists: bool
    transcript_exists: bool
    valid: bool


def _real_files(directory: Path, pattern: str) -> Iterable[Path]:
    return sorted(
        path for path in directory.glob(pattern)
        if not path.name.startswith("._") and path.name != ".DS_Store"
    )


def _read_transcripts(path: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    if not path.exists():
        return transcripts
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TRANSCRIPT_RE.match(line)
        if match:
            transcripts[match["utterance"]] = match["transcript"].strip()
    return transcripts


def build_manifest(
    dataset_root: str | Path,
    valence_lt: float = 3.0,
    activation_gt: float = 3.0,
) -> pd.DataFrame:
    root = Path(dataset_root).resolve()
    records: list[UtteranceRecord] = []

    for session in range(1, 6):
        session_root = root / f"Session{session}"
        evaluation_root = session_root / "dialog" / "EmoEvaluation"
        for evaluation_path in _real_files(evaluation_root, "*.txt"):
            dialogue_id = evaluation_path.stem
            transcript_path = session_root / "dialog" / "transcriptions" / f"{dialogue_id}.txt"
            transcripts = _read_transcripts(transcript_path)
            video_path = session_root / "dialog" / "avi" / "DivX" / f"{dialogue_id}.avi"
            marker_actor_gender = dialogue_id[5]
            scenario_type = "improvised" if "_impro" in dialogue_id else "scripted"

            for line in evaluation_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines():
                match = SUMMARY_RE.match(line)
                if not match:
                    continue
                data = match.groupdict()
                utterance_id = data["utterance"]
                speaker_gender = utterance_id.rsplit("_", 1)[1][0]
                speaker_id = f"Ses{session:02d}{speaker_gender}"
                audio_path = (
                    session_root / "sentences" / "wav" / dialogue_id / f"{utterance_id}.wav"
                )
                start, end = float(data["start"]), float(data["end"])
                emotion_code = data["emotion"]
                valence = float(data["valence"])
                activation = float(data["activation"])
                transcript = transcripts.get(utterance_id, "")
                audio_exists = audio_path.is_file()
                video_exists = video_path.is_file()
                transcript_exists = utterance_id in transcripts
                records.append(
                    UtteranceRecord(
                        utterance_id=utterance_id,
                        dialogue_id=dialogue_id,
                        session=session,
                        speaker_id=speaker_id,
                        speaker_gender=speaker_gender,
                        marker_actor_gender=marker_actor_gender,
                        scenario_type=scenario_type,
                        start_seconds=start,
                        end_seconds=end,
                        duration_seconds=max(0.0, end - start),
                        emotion_code=emotion_code,
                        emotion=EMOTION_NAMES.get(emotion_code, "unknown"),
                        emotion4=EMOTION4.get(emotion_code, ""),
                        valence=valence,
                        activation=activation,
                        dominance=float(data["dominance"]),
                        distress_proxy=valence < valence_lt and activation > activation_gt,
                        transcript=transcript,
                        audio_path=str(audio_path),
                        video_path=str(video_path),
                        evaluation_path=str(evaluation_path),
                        audio_exists=audio_exists,
                        video_exists=video_exists,
                        transcript_exists=transcript_exists,
                        valid=audio_exists and video_exists and transcript_exists and end > start,
                    )
                )

    frame = pd.DataFrame(asdict(record) for record in records)
    if frame.empty:
        raise RuntimeError(f"No IEMOCAP utterances found under {root}")
    return frame.sort_values(["session", "dialogue_id", "start_seconds"]).reset_index(drop=True)


def integrity_report(frame: pd.DataFrame) -> dict:
    duplicate_ids = frame.loc[frame.utterance_id.duplicated(), "utterance_id"].tolist()
    label_counts = Counter(frame.emotion_code)
    emotion4 = frame[frame.emotion4 != ""]
    return {
        "utterances": int(len(frame)),
        "dialogues": int(frame.dialogue_id.nunique()),
        "sessions": sorted(int(value) for value in frame.session.unique()),
        "speakers": sorted(frame.speaker_id.unique().tolist()),
        "duplicate_utterance_ids": duplicate_ids,
        "missing_audio": int((~frame.audio_exists).sum()),
        "missing_video": int((~frame.video_exists).sum()),
        "missing_transcript": int((~frame.transcript_exists).sum()),
        "invalid_rows": int((~frame.valid).sum()),
        "emotion_counts": dict(sorted(label_counts.items())),
        "emotion4_utterances": int(len(emotion4)),
        "emotion4_counts": {
            key: int(value) for key, value in emotion4.emotion4.value_counts().sort_index().items()
        },
        "distress_proxy_count": int(frame.distress_proxy.sum()),
        "duration_hours": float(frame.duration_seconds.sum() / 3600.0),
        "valence_range": [float(frame.valence.min()), float(frame.valence.max())],
        "activation_range": [float(frame.activation.min()), float(frame.activation.max())],
        "out_of_nominal_range": int(
            (
                (frame.valence < 1)
                | (frame.valence > 5)
                | (frame.activation < 1)
                | (frame.activation > 5)
                | (frame.dominance < 1)
                | (frame.dominance > 5)
            ).sum()
        ),
    }


def write_manifest(frame: pd.DataFrame, output_csv: str | Path, report_json: str | Path) -> None:
    output_csv, report_json = Path(output_csv), Path(report_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    report_json.write_text(
        json.dumps(integrity_report(frame), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
