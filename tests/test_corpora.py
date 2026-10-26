from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from iemocap_mer.corpora import (
    assign_cremad_groups,
    build_corpus_manifest,
    ravdess_actor_group,
)
from iemocap_mer.data import MultimodalFeatureDataset, multimodal_collate
from iemocap_mer.features import VideoExtractor, feature_path
from iemocap_mer.losses import concordance_correlation_loss, multitask_loss


def _probe(path: Path) -> dict:
    name = Path(path).name
    if "rate8k" in name:
        return {"duration": 1.5, "sample_rate": 8000, "error": None}
    return {"duration": 1.25, "sample_rate": 16000, "error": None}


def _extract(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"RIFF")


def test_cremad_group_cycle_continues_after_males() -> None:
    demographics = pd.DataFrame(
        {
            "ActorID": [1006, 1001, 1004, 1002, 1005, 1003, 1007],
            "Sex": ["Male", "Male", "Female", "Female", "Male", "Male", "Female"],
        }
    )
    assigned = assign_cremad_groups(demographics).set_index("ActorID")
    # Males 1001, 1003, 1005, 1006 then females 1002, 1004, 1007.
    assert assigned.loc[1001, "actor_group"] == 1
    assert assigned.loc[1003, "actor_group"] == 2
    assert assigned.loc[1005, "actor_group"] == 3
    assert assigned.loc[1006, "actor_group"] == 4
    assert assigned.loc[1002, "actor_group"] == 5
    assert assigned.loc[1004, "actor_group"] == 1
    assert assigned.loc[1007, "actor_group"] == 2
    assert assigned.loc[1001, "speaker_gender"] == "M"
    assert assigned.loc[1002, "speaker_gender"] == "F"


def test_ravdess_actor_groups() -> None:
    assert [ravdess_actor_group(actor) for actor in (1, 5, 6, 10, 11, 15, 16, 20, 21, 24)] == [
        1, 1, 2, 2, 3, 3, 4, 4, 5, 5
    ]


def test_ravdess_manifest_on_synthetic_filenames(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    names = [
        "01-01-03-01-01-01-01.mp4",
        "01-01-05-02-02-01-06.mp4",
        "01-01-01-01-01-02-21.mp4",
        "01-01-02-01-01-01-01.mp4",
        "._01-01-05-01-01-01-02.mp4",
    ]
    for name in names:
        (raw / name).write_bytes(b"mp4")
    calls = {"n": 0}

    def extract(source: Path, destination: Path) -> None:
        calls["n"] += 1
        _extract(source, destination)

    artifacts = tmp_path / "artifacts"
    report = build_corpus_manifest(
        "ravdess",
        raw,
        artifacts,
        artifacts / "manifests" / "utterances.csv",
        probe=_probe,
        extract_audio=extract,
        enforce_counts=False,
    )
    manifest_text = (artifacts / "manifests" / "utterances.csv").read_text(encoding="utf-8")
    frame = pd.read_csv(artifacts / "manifests" / "utterances.csv")
    assert len(frame) == 3
    assert calls["n"] == 3
    happy = frame.loc[frame.emotion4.eq("happy_excited")].iloc[0]
    assert happy.emotion_original == "happy"
    assert happy.emotion_code == "hap"
    assert happy.utterance_id == "rav_01-01-03-01-01-01-01"
    assert happy.dialogue_id == happy.utterance_id
    assert happy.speaker_gender == "M"
    assert int(happy.session) == 1
    assert int(happy.actor_group) == 1
    assert pd.isna(happy.valence)
    happy_line = next(line for line in manifest_text.splitlines() if happy.utterance_id in line)
    assert happy_line.endswith(",")
    female = frame.loc[frame.speaker_id.astype(str).str.zfill(2).eq("06")].iloc[0]
    assert female.speaker_gender == "F"
    assert int(female.session) == 2
    assert Path(happy.audio_path).is_file()
    build_corpus_manifest(
        "ravdess",
        raw,
        artifacts,
        artifacts / "manifests" / "utterances.csv",
        probe=_probe,
        extract_audio=extract,
        enforce_counts=False,
    )
    assert calls["n"] == 3
    groups = pd.read_csv(artifacts / "splits" / "actor_groups.csv")
    digest = (artifacts / "splits" / "actor_groups.csv.sha256").read_text(encoding="utf-8").strip()
    assert report["actor_groups_sha256"] == digest
    assert set(groups.actor_group) == {1, 2, 5}
    assert report["decode_failure_count"] == 0


def test_cremad_manifest_resamples_when_not_16k(tmp_path: Path) -> None:
    raw = tmp_path / "crema"
    (raw / "AudioWAV").mkdir(parents=True)
    (raw / "VideoFlash").mkdir()
    pd.DataFrame(
        {"ActorID": [1001, 1002], "Sex": ["Male", "Female"]}
    ).to_csv(raw / "VideoDemographics.csv", index=False)
    (raw / "AudioWAV" / "1001_DFA_HAP_XX.wav").write_bytes(b"wav")
    (raw / "VideoFlash" / "1001_DFA_HAP_XX.flv").write_bytes(b"flv")
    (raw / "AudioWAV" / "1002_DFA_ANG_XX.rate8k.wav")
    # Filename must match Actor_Sentence_EMO_Level; encode the rate in a sidecar probe hook.
    (raw / "AudioWAV" / "1002_DFA_SAD_XX.wav").write_bytes(b"wav-8k")
    (raw / "VideoFlash" / "1002_DFA_SAD_XX.flv").write_bytes(b"flv")
    (raw / "AudioWAV" / "1001_DFA_DIS_XX.wav").write_bytes(b"skip")

    def probe(path: Path) -> dict:
        if Path(path).name == "1002_DFA_SAD_XX.wav":
            return {"duration": 2.0, "sample_rate": 8000, "error": None}
        return _probe(path)

    extracted = []

    def extract(source: Path, destination: Path) -> None:
        extracted.append(Path(source).name)
        _extract(source, destination)

    artifacts = tmp_path / "artifacts"
    build_corpus_manifest(
        "cremad",
        raw,
        artifacts,
        artifacts / "manifests" / "utterances.csv",
        probe=probe,
        extract_audio=extract,
        enforce_counts=False,
    )
    frame = pd.read_csv(artifacts / "manifests" / "utterances.csv")
    assert list(frame.emotion4) == ["happy_excited", "sad"] or set(frame.emotion4) == {
        "happy_excited",
        "sad",
    }
    happy = frame.loc[frame.utterance_id.eq("crd_1001_DFA_HAP_XX")].iloc[0]
    assert happy.emotion_original == "happy"
    assert "AudioWAV" in happy.audio_path
    assert extracted == ["1002_DFA_SAD_XX.wav"]
    sad = frame.loc[frame.utterance_id.eq("crd_1002_DFA_SAD_XX")].iloc[0]
    assert "audio_wav" in sad.audio_path
    assert int(happy.session) == 1
    assert int(sad.session) == 2
    assert bool(happy.valid) and bool(sad.valid)


def test_dataset_and_loss_skip_va_when_unavailable(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "utterance_id": ["crd_1001_DFA_HAP_XX"],
            "emotion4": ["happy_excited"],
            "session": [1],
        }
    )
    dataset = MultimodalFeatureDataset(
        frame,
        ["crd_1001_DFA_HAP_XX"],
        tmp_path,
        load_audio=False,
        load_video=False,
        has_va=False,
    )
    item = dataset[0]
    assert item["has_va"] is False
    assert torch.equal(item["va"], torch.zeros(2))
    batch = multimodal_collate([item, item])
    assert batch["has_va"] is False
    logits = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    labels = torch.tensor([0, 1])
    output = {"logits": logits, "va": torch.ones(2, 2)}
    weights = {"emotion": 1.0, "va": 1.0, "alignment": 0.0}

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("concordance loss must not be computed")

    original = concordance_correlation_loss
    try:
        import iemocap_mer.losses as losses

        losses.concordance_correlation_loss = _forbidden
        result = multitask_loss(output, labels, batch["va"], weights, has_va=False)
    finally:
        losses.concordance_correlation_loss = original
    assert "va" not in result
    assert torch.allclose(result["total"], F.cross_entropy(logits, labels))


def test_feature_path_keeps_iemocap_session_directory() -> None:
    path = feature_path("/tmp/features", "audio", "Ses01F_impro01_F000", feature_dir="audio_w2v")
    assert path.as_posix().endswith("audio_w2v/Session1/Ses01F_impro01_F000.pt")
    other = feature_path(
        "/tmp/features", "audio", "rav_01-01-03-01-01-01-01", feature_dir="audio_w2v", session=4
    )
    assert other.as_posix().endswith("audio_w2v/Session4/rav_01-01-03-01-01-01-01.pt")


def test_full_frame_returns_the_whole_frame() -> None:
    frame = np.zeros((4, 8, 3), dtype=np.uint8)
    frame[:, :4] = 1
    frame[:, 4:] = 2
    half = VideoExtractor(face_region="speaker_half")
    full = VideoExtractor(face_region="full_frame")
    female = half._speaker_half(frame, "F")
    male = half._speaker_half(frame, "M")
    assert female.shape == (4, 4, 3)
    assert male.shape == (4, 4, 3)
    assert int(female[0, 0, 0]) == 1
    assert int(male[0, 0, 0]) == 2
    assert np.array_equal(full._speaker_half(frame, "F"), frame)
    assert np.array_equal(full._speaker_half(frame, "M"), frame)


def test_au_extractor_full_frame_uses_cropper(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyDetector:
        def __init__(self, **_kwargs) -> None:
            pass

    monkeypatch.setattr("feat.Detectorv1", DummyDetector)
    from iemocap_mer.features_au import VideoAUExtractor

    extractor = VideoAUExtractor(face_region="full_frame")
    frame = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    assert np.array_equal(extractor._speaker_half(frame, "M"), frame)
    assert extractor.cropper.face_region == "full_frame"
