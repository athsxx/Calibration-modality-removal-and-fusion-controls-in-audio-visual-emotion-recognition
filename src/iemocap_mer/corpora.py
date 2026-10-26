"""RAVDESS and CREMA-D manifests in the IEMOCAP column layout."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import pandas as pd

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "ffprobe")

RAVDESS_NAME = re.compile(
    r"^01-01-(?P<emotion>01|03|04|05)-(?P<intensity>\d{2})-"
    r"(?P<statement>\d{2})-(?P<repetition>\d{2})-(?P<actor>\d{2})\.mp4$"
)
CREMA_NAME = re.compile(
    r"^(?P<actor>\d{4})_(?P<sentence>[A-Z]{3})_(?P<emotion>ANG|HAP|NEU|SAD)_(?P<level>[A-Z]{2})$"
)

# emotion4, emotion_code, emotion_original. Class 1 stays happy_excited so
# LABEL_TO_INDEX is unchanged; emotion_original records the source word "happy".
RAVDESS_LABELS = {
    "01": ("neutral", "neu", "neutral"),
    "03": ("happy_excited", "hap", "happy"),
    "04": ("sad", "sad", "sad"),
    "05": ("angry", "ang", "angry"),
}
CREMA_LABELS = {
    "ANG": ("angry", "ang", "angry"),
    "HAP": ("happy_excited", "hap", "happy"),
    "NEU": ("neutral", "neu", "neutral"),
    "SAD": ("sad", "sad", "sad"),
}

MANIFEST_COLUMNS = [
    "utterance_id",
    "dialogue_id",
    "session",
    "speaker_id",
    "speaker_gender",
    "actor_group",
    "start_seconds",
    "end_seconds",
    "duration_seconds",
    "emotion4",
    "emotion_original",
    "emotion_code",
    "valence",
    "activation",
    "dominance",
    "audio_path",
    "video_path",
    "audio_exists",
    "video_exists",
    "valid",
    "transcript",
]

RAVDESS_CLASS_COUNTS = {"angry": 8, "happy_excited": 8, "neutral": 4, "sad": 8}
CREMA_CLASS_COUNTS = {"angry": 1271, "happy_excited": 1271, "neutral": 1087, "sad": 1271}


class CorpusCountError(ValueError):
    """Manifest row or class counts do not match the locked protocol."""


def probe_media(path: Path, *, ffprobe: str = FFPROBE) -> dict:
    """Return duration and, for audio, sample rate. Never raises."""
    if not Path(path).is_file():
        return {"duration": None, "sample_rate": None, "error": "missing"}
    try:
        with Path(path).open("rb") as handle:
            head = handle.read(40)
    except OSError as exc:
        return {"duration": None, "sample_rate": None, "error": str(exc)}
    if head.startswith(b"version https://git-lfs"):
        return {
            "duration": None,
            "sample_rate": None,
            "error": "git-lfs pointer; media file was not checked out",
        }
    duration_run = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    rate_run = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    duration = _positive_float(duration_run.stdout)
    sample_rate = _positive_int(rate_run.stdout)
    error = None
    if duration_run.returncode != 0 or duration is None:
        detail = (duration_run.stderr or rate_run.stderr or "ffprobe failed").strip()
        error = detail.splitlines()[-1] if detail else "ffprobe failed"
    return {"duration": duration, "sample_rate": sample_rate, "error": error}


def extract_wav(source: Path, destination: Path, *, ffmpeg: str = FFMPEG) -> None:
    """Write 16 kHz mono PCM WAV atomically. Skip work when destination is ready."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The temporary name must end in .wav. ffmpeg picks the muxer from the suffix,
    # and ".partial" is rejected ("Error opening output files: Invalid argument").
    temporary = destination.with_name(f".{destination.stem}.partial.wav")
    completed = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(temporary),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not temporary.is_file():
        if temporary.exists():
            temporary.unlink()
        detail = (completed.stderr or "ffmpeg failed").strip()
        raise RuntimeError(detail.splitlines()[-1] if detail else "ffmpeg failed")
    os.replace(temporary, destination)


def ravdess_actor_group(actor: int) -> int:
    if 1 <= actor <= 5:
        return 1
    if 6 <= actor <= 10:
        return 2
    if 11 <= actor <= 15:
        return 3
    if 16 <= actor <= 20:
        return 4
    if 21 <= actor <= 24:
        return 5
    raise ValueError(f"RAVDESS actor out of range: {actor}")


def assign_cremad_groups(demographics: pd.DataFrame) -> pd.DataFrame:
    """Sex-balanced groups from protocol v3.

    Male ActorIDs sorted ascending take groups 1,2,3,4,5,1,... Female ActorIDs
    sorted ascending continue that cycle; they do not restart at group 1.
    """
    frame = demographics.copy()
    frame["ActorID"] = frame["ActorID"].astype(int)
    sex = frame["Sex"].astype(str).str.strip().str.lower()
    frame["speaker_gender"] = sex.map({"male": "M", "female": "F"})
    if frame["speaker_gender"].isna().any():
        unknown = sorted(frame.loc[frame.speaker_gender.isna(), "Sex"].astype(str).unique())
        raise ValueError(f"VideoDemographics.csv has sex values other than Male/Female: {unknown}")
    males = sorted(frame.loc[frame.speaker_gender.eq("M"), "ActorID"].unique())
    females = sorted(frame.loc[frame.speaker_gender.eq("F"), "ActorID"].unique())
    mapping: dict[int, int] = {}
    group = 1
    for actor in list(males) + list(females):
        mapping[int(actor)] = group
        group = 1 if group == 5 else group + 1
    assigned = frame.drop_duplicates("ActorID").copy()
    assigned["actor_group"] = assigned["ActorID"].map(mapping).astype(int)
    return assigned.sort_values("ActorID").reset_index(drop=True)


def build_corpus_manifest(
    corpus: str,
    raw_dir: str | Path,
    artifacts_root: str | Path,
    manifest_path: str | Path,
    *,
    probe=probe_media,
    extract_audio=extract_wav,
    enforce_counts: bool = True,
) -> dict:
    corpus = corpus.lower()
    raw_dir = Path(raw_dir)
    artifacts_root = Path(artifacts_root)
    manifest_path = Path(manifest_path)
    if corpus == "ravdess":
        frame, groups, failures = _build_ravdess(
            raw_dir, artifacts_root, probe=probe, extract_audio=extract_audio
        )
    elif corpus == "cremad":
        frame, groups, failures = _build_cremad(
            raw_dir, artifacts_root, probe=probe, extract_audio=extract_audio
        )
    else:
        raise ValueError(f"Unknown corpus: {corpus}")

    count_error = None
    if enforce_counts:
        try:
            if corpus == "ravdess":
                _validate_ravdess_counts(frame)
            else:
                _validate_cremad_counts(frame)
        except CorpusCountError as exc:
            count_error = str(exc)

    _atomic_csv(frame[MANIFEST_COLUMNS], manifest_path)
    groups_path = artifacts_root / "splits" / "actor_groups.csv"
    _atomic_csv(groups, groups_path)
    digest = hashlib.sha256(groups_path.read_bytes()).hexdigest()
    _atomic_text(groups_path.with_suffix(".csv.sha256"), digest + "\n")
    report = _report(corpus, frame, failures, digest)
    if count_error:
        report["counts_ok"] = False
        report["count_error"] = count_error
    report_path = artifacts_root / "reports" / "manifest_report.json"
    _atomic_text(report_path, json.dumps(report, indent=2) + "\n")
    if count_error:
        raise CorpusCountError(count_error)
    return report


def _build_ravdess(raw_dir: Path, artifacts_root: Path, *, probe, extract_audio) -> tuple:
    rows = []
    failures = []
    for path in sorted(raw_dir.glob("*.mp4")):
        if path.name.startswith("._"):
            continue
        match = RAVDESS_NAME.match(path.name)
        if match is None:
            continue
        actor = int(match.group("actor"))
        emotion4, emotion_code, emotion_original = RAVDESS_LABELS[match.group("emotion")]
        group = ravdess_actor_group(actor)
        utterance_id = "rav_" + path.stem
        audio_path = artifacts_root / "audio_wav" / f"{utterance_id}.wav"
        audio_error = _ensure_wav(path, audio_path, probe=probe, extract_audio=extract_audio)
        video_info = probe(path)
        duration = video_info["duration"]
        if audio_error:
            failures.append({"utterance_id": utterance_id, "file": str(audio_path), "error": audio_error})
        if video_info["error"] or not duration or duration <= 0:
            failures.append(
                {
                    "utterance_id": utterance_id,
                    "file": str(path),
                    "error": video_info["error"] or "duration not positive",
                }
            )
            duration = duration if duration and duration > 0 else float("nan")
        audio_ok = audio_error is None and audio_path.is_file()
        video_ok = path.is_file() and bool(video_info["duration"] and video_info["duration"] > 0)
        rows.append(
            _row(
                utterance_id=utterance_id,
                session=group,
                speaker_id=f"{actor:02d}",
                speaker_gender="M" if actor % 2 else "F",
                duration=duration,
                emotion4=emotion4,
                emotion_original=emotion_original,
                emotion_code=emotion_code,
                audio_path=str(audio_path),
                video_path=str(path.resolve()),
                audio_exists=audio_path.is_file(),
                video_exists=path.is_file(),
                valid=audio_ok and video_ok,
            )
        )
    frame = _as_manifest(rows)
    groups = (
        frame[["speaker_id", "speaker_gender", "actor_group"]]
        .drop_duplicates()
        .rename(columns={"speaker_id": "actor_id"})
        .sort_values("actor_id")
        .reset_index(drop=True)
    )
    return frame, groups, failures


def _build_cremad(raw_dir: Path, artifacts_root: Path, *, probe, extract_audio) -> tuple:
    demographics = pd.read_csv(raw_dir / "VideoDemographics.csv")
    assigned = assign_cremad_groups(demographics)
    gender = assigned.set_index("ActorID")["speaker_gender"].to_dict()
    groups_by_actor = assigned.set_index("ActorID")["actor_group"].to_dict()
    rows = []
    failures = []
    audio_root = raw_dir / "AudioWAV"
    video_root = raw_dir / "VideoFlash"
    stems = []
    for path in sorted(audio_root.glob("*.wav")):
        if path.name.startswith("._"):
            continue
        if CREMA_NAME.match(path.stem):
            stems.append(path.stem)
    for stem in stems:
        match = CREMA_NAME.match(stem)
        actor = int(match.group("actor"))
        emotion4, emotion_code, emotion_original = CREMA_LABELS[match.group("emotion")]
        if actor not in groups_by_actor:
            failures.append({"utterance_id": "crd_" + stem, "file": stem, "error": "actor missing from demographics"})
            continue
        utterance_id = "crd_" + stem
        source_audio = audio_root / f"{stem}.wav"
        video_path = video_root / f"{stem}.flv"
        audio_info = probe(source_audio)
        if (
            audio_info["error"] is None
            and audio_info["sample_rate"] == 16000
            and audio_info["duration"]
            and audio_info["duration"] > 0
        ):
            audio_path = source_audio
            audio_error = None
            duration = audio_info["duration"]
        else:
            audio_path = artifacts_root / "audio_wav" / f"{utterance_id}.wav"
            if audio_info["error"] is None and audio_info["duration"] and audio_info["duration"] > 0:
                audio_error = _ensure_wav(source_audio, audio_path, probe=probe, extract_audio=extract_audio)
                probed = probe(audio_path) if audio_error is None else {"duration": None, "error": audio_error}
                duration = probed.get("duration") or audio_info["duration"]
                if audio_error is None and (probed.get("sample_rate") not in (None, 16000) or not probed.get("duration")):
                    audio_error = probed.get("error") or "resampled wav is not 16 kHz"
            else:
                audio_error = audio_info["error"] or "duration not positive"
                duration = float("nan")
        video_info = probe(video_path)
        if video_info["duration"] and video_info["duration"] > 0:
            duration = video_info["duration"]
        elif not (isinstance(duration, float) and duration > 0):
            duration = float("nan")
        if audio_error:
            failures.append({"utterance_id": utterance_id, "file": str(audio_path), "error": audio_error})
        if video_info["error"] or not (video_info["duration"] and video_info["duration"] > 0):
            failures.append(
                {
                    "utterance_id": utterance_id,
                    "file": str(video_path),
                    "error": video_info["error"] or "duration not positive",
                }
            )
        audio_ok = audio_error is None and Path(audio_path).is_file()
        video_ok = video_path.is_file() and bool(video_info["duration"] and video_info["duration"] > 0)
        rows.append(
            _row(
                utterance_id=utterance_id,
                session=int(groups_by_actor[actor]),
                speaker_id=f"{actor:04d}",
                speaker_gender=gender[actor],
                duration=duration,
                emotion4=emotion4,
                emotion_original=emotion_original,
                emotion_code=emotion_code,
                audio_path=str(Path(audio_path).resolve()),
                video_path=str(video_path.resolve()),
                audio_exists=Path(audio_path).is_file(),
                video_exists=video_path.is_file(),
                valid=audio_ok and video_ok,
            )
        )
    frame = _as_manifest(rows)
    present = set(frame.speaker_id.astype(str))
    groups = assigned.rename(columns={"ActorID": "actor_id"}).copy()
    groups["actor_id"] = groups["actor_id"].map(lambda value: f"{int(value):04d}")
    groups = groups.loc[groups.actor_id.isin(present), ["actor_id", "speaker_gender", "actor_group"]]
    groups = groups.sort_values("actor_id").reset_index(drop=True)
    return frame, groups, failures


def _ensure_wav(source: Path, destination: Path, *, probe, extract_audio) -> str | None:
    if destination.is_file():
        info = probe(destination)
        if (
            info["error"] is None
            and info["sample_rate"] == 16000
            and info["duration"]
            and info["duration"] > 0
        ):
            return None
    try:
        extract_audio(source, destination)
    except Exception as exc:
        return str(exc)
    info = probe(destination)
    if info["error"] or not info["duration"] or info["duration"] <= 0 or info["sample_rate"] != 16000:
        return info["error"] or "extracted wav is not 16 kHz with a positive duration"
    return None


def _row(**fields) -> dict:
    duration = fields["duration"]
    try:
        duration_value = float(duration)
    except (TypeError, ValueError):
        duration_value = float("nan")
    return {
        "utterance_id": fields["utterance_id"],
        "dialogue_id": fields["utterance_id"],
        "session": int(fields["session"]),
        "speaker_id": fields["speaker_id"],
        "speaker_gender": fields["speaker_gender"],
        "actor_group": int(fields["session"]),
        "start_seconds": 0.0,
        "end_seconds": duration_value,
        "duration_seconds": duration_value,
        "emotion4": fields["emotion4"],
        "emotion_original": fields["emotion_original"],
        "emotion_code": fields["emotion_code"],
        "valence": float("nan"),
        "activation": float("nan"),
        "dominance": float("nan"),
        "audio_path": fields["audio_path"],
        "video_path": fields["video_path"],
        "audio_exists": bool(fields["audio_exists"]),
        "video_exists": bool(fields["video_exists"]),
        "valid": bool(fields["valid"]),
        "transcript": "",
    }


def _as_manifest(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if frame.empty:
        return frame
    return frame.sort_values("utterance_id").reset_index(drop=True)


def _validate_ravdess_counts(frame: pd.DataFrame) -> None:
    if len(frame) != 672:
        raise CorpusCountError(f"RAVDESS manifest has {len(frame)} rows, expected 672")
    per_actor = frame.groupby("speaker_id").size()
    if not (per_actor == 28).all():
        raise CorpusCountError(f"RAVDESS actors do not all have 28 clips: {per_actor.to_dict()}")
    for speaker_id, sub in frame.groupby("speaker_id"):
        counts = sub.emotion4.value_counts().to_dict()
        if counts != RAVDESS_CLASS_COUNTS:
            raise CorpusCountError(f"RAVDESS actor {speaker_id} class counts {counts}")


def _validate_cremad_counts(frame: pd.DataFrame) -> None:
    if len(frame) != 4900:
        raise CorpusCountError(f"CREMA-D manifest has {len(frame)} rows, expected 4900")
    counts = frame.emotion4.value_counts().to_dict()
    if counts != CREMA_CLASS_COUNTS:
        raise CorpusCountError(f"CREMA-D class counts {counts}, expected {CREMA_CLASS_COUNTS}")


def _counts_ok(corpus: str, frame: pd.DataFrame) -> bool:
    try:
        if corpus == "ravdess":
            _validate_ravdess_counts(frame)
        else:
            _validate_cremad_counts(frame)
    except CorpusCountError:
        return False
    return True


def _report(corpus: str, frame: pd.DataFrame, failures: list[dict], digest: str) -> dict:
    def _sized(series: pd.Series) -> dict:
        return {str(key): int(value) for key, value in series.items()}

    per_group_gender = {}
    if not frame.empty:
        for session, sub in frame.groupby("session"):
            per_group_gender[str(int(session))] = {
                str(gender): int(count) for gender, count in sub.speaker_gender.value_counts().items()
            }
    speakers = {}
    if not frame.empty:
        speakers = {
            str(int(session)): int(sub.speaker_id.nunique())
            for session, sub in frame.groupby("session")
        }
    return {
        "corpus": corpus,
        "rows": int(len(frame)),
        "valid_rows": int(frame.valid.sum()) if not frame.empty else 0,
        "per_class": _sized(frame.emotion4.value_counts()) if not frame.empty else {},
        "per_group": _sized(frame.session.value_counts()) if not frame.empty else {},
        "speakers_per_group": speakers,
        "gender_counts": _sized(frame.speaker_gender.value_counts()) if not frame.empty else {},
        "gender_per_group": per_group_gender,
        "per_actor": _sized(frame.groupby("speaker_id").size()) if not frame.empty else {},
        "decode_failures": failures,
        "decode_failure_count": len(failures),
        "actor_groups_sha256": digest,
        "counts_ok": _counts_ok(corpus, frame),
    }


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _positive_float(text: str) -> float | None:
    try:
        value = float(text.strip())
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _positive_int(text: str) -> int | None:
    try:
        value = int(text.strip())
    except ValueError:
        return None
    return value if value > 0 else None
