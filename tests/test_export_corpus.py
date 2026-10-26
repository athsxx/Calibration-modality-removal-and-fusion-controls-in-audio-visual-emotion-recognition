"""Round-trip the corpus exporter on a tiny checkpoint. No corpus audio is read."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "mer_companion"))

import mtap_export_corpus as exporter
from iemocap_mer.analysis import load_trained_model
from iemocap_mer.data import multimodal_collate
from iemocap_mer.selective_fusion import PREDICTION_COLUMNS
from iemocap_mer.training import build_model, forward_model, move_batch
from merkit.analysis import classification, softmax, validate_table
from merkit.export import alter_batch


def _manifest_rows() -> list[dict]:
    rows = []
    plan = {
        "train": [("a1", "spkA", "angry"), ("a2", "spkA", "sad")],
        "dev": [("b1", "spkB", "happy_excited"), ("b2", "spkB", "neutral")],
        "test": [
            ("c1", "spkC", "angry"),
            ("c2", "spkC", "happy"),
            ("c3", "spkC", "neutral"),
            ("c4", "spkC", "sad"),
        ],
    }
    for _part, items in plan.items():
        for utterance_id, speaker, emotion in items:
            rows.append({
                "utterance_id": utterance_id,
                "session": 1,
                "speaker_id": speaker,
                "emotion4": emotion,
            })
    return rows


def _write_features(root: Path, cache: str, rows: list[dict], width: int) -> None:
    for row in rows:
        if row["utterance_id"].startswith("a"):
            continue
        path = root / "features" / cache / "Session1" / f"{row['utterance_id']}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator().manual_seed(abs(hash(row["utterance_id"] + cache)) % (2**31))
        torch.save(torch.randn(6, width, generator=generator), path)


def _checkpoint(path: Path, model_name: str, audio_dim: int, video_dim: int, audio_dir: str, video_dir: str) -> None:
    config = {"hidden_dim": 8, "num_classes": 4, "dropout": 0.0}
    model = build_model(model_name, audio_dim, video_dim, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "model_name": model_name,
        "audio_dim": audio_dim,
        "video_dim": video_dim,
        "audio_dir": audio_dir,
        "video_dir": video_dir,
        "model_config": config,
        "epoch": 1,
        "stage": 1,
        "audio_max_tokens": 4,
        "video_max_tokens": 4,
    }, path)


def _clean_predictions(artifacts: Path, relative: str, frame: pd.DataFrame, ids: list[str], model_name: str,
                       audio_dir: str, video_dir: str) -> pd.DataFrame:
    device = torch.device("cpu")
    model, _checkpoint_payload = load_trained_model(artifacts / relative, device)
    dataset = exporter.StrictCorpusFeatures(
        frame, ids, artifacts / "features",
        {"model_name": model_name, "audio_dir": audio_dir, "video_dir": video_dir},
        4, 4,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=multimodal_collate)
    stored = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = alter_batch(move_batch(raw_batch, device), "clean", exporter.CORRUPTION_SEED)
            logits = forward_model(model, model_name, batch)["logits"].detach().cpu().double().numpy()
            labels = raw_batch["label"].numpy()
            for index, utterance_id in enumerate(raw_batch["utterance_id"]):
                stored.append({
                    "utterance_id": utterance_id,
                    "label": int(labels[index]),
                    "prediction": int(logits[index].argmax()),
                })
    return pd.DataFrame(stored)


def _build(tmp_path: Path) -> tuple[Path, float]:
    rows = _manifest_rows()
    artifacts = tmp_path / "artifacts_tiny"
    _write_features(artifacts, "audio_tiny", rows, 4)
    _write_features(artifacts, "video_tiny", rows, 4)
    (artifacts / "manifests").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(artifacts / "manifests" / "utterances.csv", index=False)
    splits = {
        "fold_1": {
            "train": ["a1", "a2"],
            "dev": ["b1", "b2"],
            "test": ["c1", "c2", "c3", "c4"],
        }
    }
    (artifacts / "splits").mkdir(parents=True, exist_ok=True)
    (artifacts / "splits" / "folds.json").write_text(json.dumps(splits), encoding="utf-8")
    _checkpoint(artifacts / "checkpoints" / "audio_tiny.pt", "audio", 4, 1, "audio_tiny", "video_tiny")
    _checkpoint(artifacts / "checkpoints" / "video_tiny.pt", "video", 1, 4, "audio_tiny", "video_tiny")
    frame = pd.read_csv(artifacts / "manifests" / "utterances.csv", dtype={"utterance_id": str})
    audio_reference = _clean_predictions(
        artifacts, "checkpoints/audio_tiny.pt", frame, ["c1", "c2", "c3", "c4"],
        "audio", "audio_tiny", "video_tiny",
    )
    audio_f1 = classification(
        audio_reference.label.to_numpy(), audio_reference.prediction.to_numpy(),
    )["macro_f1"]
    video_reference = _clean_predictions(
        artifacts, "checkpoints/video_tiny.pt", frame, ["c1", "c2", "c3", "c4"],
        "video", "audio_tiny", "video_tiny",
    )
    video_f1 = classification(
        video_reference.label.to_numpy(), video_reference.prediction.to_numpy(),
    )["macro_f1"]
    audio_reference.to_csv(artifacts / "checkpoints" / "audio_tiny.predictions.csv", index=False)
    (artifacts / "checkpoints" / "audio_tiny.metrics.json").write_text(json.dumps({
        "best_epoch": 1,
        "seed": 0,
        "test": {"macro_f1": audio_f1},
        "aborted_nonfinite": False,
        "nonfinite_batch_count": 0,
    }), encoding="utf-8")
    (artifacts / "checkpoints" / "video_tiny.metrics.json").write_text(json.dumps({
        "status": "complete",
        "best_epoch": 1,
        "seed": 0,
        "test": {"macro_f1": video_f1},
        "aborted_nonfinite": False,
        "nonfinite_batch_count": 0,
    }), encoding="utf-8")
    spec = tmp_path / "runs.csv"
    spec.write_text(
        "system,checkpoint,seed,fold\n"
        "audio_w2v,checkpoints/audio_tiny.pt,0,1\n"
        "video_au,checkpoints/video_tiny.pt,0,1\n",
        encoding="utf-8",
    )
    return artifacts, audio_f1


def test_export_contract_round_trip_and_history_flag(tmp_path: Path) -> None:
    artifacts, _audio_f1 = _build(tmp_path)
    out = tmp_path / "export"
    code = exporter.export_corpus(exporter.parser().parse_args([
        "--artifacts-root", str(artifacts),
        "--manifest", "manifests/utterances.csv",
        "--splits", "splits/folds.json",
        "--dataset", "tiny",
        "--run-spec", str(tmp_path / "runs.csv"),
        "--out", str(out),
        "--device", "cpu",
        "--batch-size", "2",
    ]))
    assert code == 0
    manifest = json.loads((out / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["dataset"] == "tiny"
    assert manifest["history_unknown_count"] == 1
    assert len(manifest["runs"]) == 2
    split = pd.read_csv(out / "split_manifest.csv")
    assert list(split.columns) == [
        "dataset", "fold", "split", "utterance_id", "session", "speaker_id", "label",
    ]
    by_model = {run["model"]: run for run in manifest["runs"]}
    assert by_model["audio_w2v"]["history_status"] == "history_unknown"
    assert by_model["audio_w2v"]["parity_absolute_difference"] == 0.0
    assert "exact labels" in by_model["audio_w2v"]["clean_parity_basis"]
    assert by_model["video_au"]["history_status"] == "eligible"
    assert by_model["video_au"]["parity_absolute_difference"] == 0.0
    audio_test = pd.read_csv(
        out / "runs" / "audio_tiny" / "test_predictions.csv.gz", dtype={"utterance_id": str},
    )
    video_test = pd.read_csv(
        out / "runs" / "video_tiny" / "test_predictions.csv.gz", dtype={"utterance_id": str},
    )
    audio_dev = pd.read_csv(
        out / "runs" / "audio_tiny" / "validation_predictions.csv.gz", dtype={"utterance_id": str},
    )
    assert list(audio_test.columns) == PREDICTION_COLUMNS
    assert set(audio_dev.condition) == {"clean"}
    assert set(audio_test.condition) == {"clean", "zero_audio"}
    assert set(video_test.condition) == {
        "clean", "zero_video", "video_token_loss_25", "video_token_loss_50", "video_token_loss_75",
    }
    validate_table(audio_dev, "validation")
    validate_table(audio_test, "test")
    validate_table(video_test, "test")
    zero = video_test[video_test.condition == "zero_video"]
    assert set(zero.video_kept_tokens.astype(int))  # mask is unchanged by zero-fill
    clean = video_test[video_test.condition == "clean"].sort_values("utterance_id")
    device = torch.device("cpu")
    model, _payload = load_trained_model(artifacts / "checkpoints" / "video_tiny.pt", device)
    frame = pd.read_csv(artifacts / "manifests" / "utterances.csv", dtype={"utterance_id": str})
    dataset = exporter.StrictCorpusFeatures(
        frame, ["c1"], artifacts / "features",
        {"model_name": "video", "audio_dir": "audio_tiny", "video_dir": "video_tiny"},
        4, 4,
    )
    batch = move_batch(multimodal_collate([dataset[0]]), device)
    with torch.inference_mode():
        logits = forward_model(model, "video", batch)["logits"].detach().cpu().double().numpy()
    exported = clean[clean.utterance_id == "c1"][[f"z_{k}" for k in range(4)]].to_numpy(dtype=float)
    assert np.allclose(exported, logits)
    assert np.allclose(softmax(exported), clean[clean.utterance_id == "c1"][[f"p_{k}" for k in range(4)]].to_numpy())
    resume = exporter.export_corpus(exporter.parser().parse_args([
        "--artifacts-root", str(artifacts),
        "--manifest", "manifests/utterances.csv",
        "--splits", "splits/folds.json",
        "--dataset", "tiny",
        "--run-spec", str(tmp_path / "runs.csv"),
        "--out", str(out),
        "--device", "cpu",
        "--batch-size", "2",
        "--resume",
    ]))
    assert resume == 0
    try:
        exporter.export_corpus(exporter.parser().parse_args([
            "--artifacts-root", str(artifacts),
            "--manifest", "manifests/utterances.csv",
            "--splits", "splits/folds.json",
            "--dataset", "tiny",
            "--run-spec", str(tmp_path / "runs.csv"),
            "--out", str(out),
            "--device", "cpu",
        ]))
    except OSError as exc:
        assert "exist" in str(exc).lower()
    else:
        raise AssertionError("A second export must refuse to overwrite")


def test_parity_failure_is_detected(tmp_path: Path) -> None:
    artifacts, _audio_f1 = _build(tmp_path)
    reference = pd.read_csv(artifacts / "checkpoints" / "audio_tiny.predictions.csv", dtype={"utterance_id": str})
    reference.loc[0, "prediction"] = int((int(reference.loc[0, "prediction"]) + 1) % 4)
    reference.to_csv(artifacts / "checkpoints" / "audio_tiny.predictions.csv", index=False)
    spec = tmp_path / "audio_only.csv"
    spec.write_text("system,checkpoint,seed,fold\naudio_w2v,checkpoints/audio_tiny.pt,0,1\n", encoding="utf-8")
    out = tmp_path / "bad_export"
    code = exporter.export_corpus(exporter.parser().parse_args([
        "--artifacts-root", str(artifacts),
        "--manifest", "manifests/utterances.csv",
        "--splits", "splits/folds.json",
        "--dataset", "tiny",
        "--run-spec", str(spec),
        "--out", str(out),
        "--device", "cpu",
    ]))
    assert code == 2
    failure = json.loads((out / "runs" / "audio_tiny" / "failure.json").read_text(encoding="utf-8"))
    assert "differ" in failure["error"].lower() or "mismatch" in failure["error"].lower()
    assert not (out / "runs" / "audio_tiny" / "complete.json").exists()


def test_eligibility_rules() -> None:
    status, rejected, unknown = exporter.assess_eligibility(
        {"test": {"macro_f1": 0.5}}, "audio", {"stage": 1},
    )
    assert status == "history_unknown"
    assert unknown and not rejected
    status, rejected, _unknown = exporter.assess_eligibility(
        {"status": "complete", "stage2_reached": True, "used_stage1_fallback": True},
        "essence",
        {"stage": 2},
    )
    assert status == "rejected"
    assert any("used_stage1_fallback" in reason for reason in rejected)
    status, rejected, _unknown = exporter.assess_eligibility(
        {"status": "failed", "stage2_reached": False, "used_stage1_fallback": False},
        "essence",
        {"stage": 1},
    )
    assert status == "rejected"


def test_duplicate_ids_and_nonfinite_features(tmp_path: Path) -> None:
    frame = pd.DataFrame([
        {"utterance_id": "a", "session": 1, "speaker_id": "s", "emotion4": "angry"},
        {"utterance_id": "b", "session": 1, "speaker_id": "t", "emotion4": "sad"},
    ])
    folds = {"fold_1": {"train": ["a", "a"], "dev": ["b"], "test": ["b"]}}
    try:
        exporter.split_records("tiny", frame, folds)
    except ValueError as exc:
        assert "Duplicate" in str(exc) or "leakage" in str(exc)
    else:
        raise AssertionError("duplicate split ids must fail")
    path = tmp_path / "bad.pt"
    torch.save(torch.tensor([[float("nan"), 1.0], [0.0, 0.0]]), path)
    try:
        exporter.load_feature(path, 4)
    except ValueError as exc:
        assert "Invalid feature" in str(exc)
    else:
        raise AssertionError("non-finite features must fail")


def test_manifest_does_not_need_valence(tmp_path: Path) -> None:
    artifacts, _audio_f1 = _build(tmp_path)
    frame = pd.read_csv(artifacts / "manifests" / "utterances.csv")
    assert "valence" not in frame.columns and "activation" not in frame.columns
    dataset = exporter.StrictCorpusFeatures(
        frame.assign(utterance_id=frame.utterance_id.astype(str)),
        ["b1"],
        artifacts / "features",
        {"model_name": "audio", "audio_dir": "audio_tiny", "video_dir": "video_tiny"},
        4, 4,
    )
    item = dataset[0]
    assert item["has_va"] is False
    assert torch.isfinite(item["audio"]).all()
    assert item["label"] == 1
