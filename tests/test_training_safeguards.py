import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from iemocap_mer.data import MultimodalFeatureDataset, multimodal_collate
from iemocap_mer.features import feature_path
from iemocap_mer.training import (
    CheckpointExistsError,
    Stage2NotReached,
    evaluate_loader,
    run_training,
)

METRIC_FIELDS = (
    "run_id",
    "dataset_name",
    "status",
    "nonfinite_batch_count",
    "aborted_nonfinite",
    "stage2_reached",
    "used_stage1_fallback",
    "config_sha256",
    "git_commit",
    "wall_clock_seconds",
    "seed",
    "fold",
    "model",
)


def _frame_and_features(root: Path) -> tuple[pd.DataFrame, dict]:
    labels = ["angry", "happy_excited", "neutral", "sad"]
    rows = []
    parts = {"train": [], "dev": [], "test": []}
    for index, (part, session) in enumerate((("train", 1), ("train", 1), ("dev", 2), ("test", 3))):
        utterance_id = f"Ses{session:02d}F_clip_{index}"
        rows.append(
            {
                "utterance_id": utterance_id,
                "emotion4": labels[index],
                "valence": 3.0,
                "activation": 3.0,
                "session": session,
                "speaker_id": f"spk{session}",
            }
        )
        parts[part].append(utterance_id)
        for modality in ("audio", "video"):
            path = feature_path(root, modality, utterance_id, session=session)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.zeros(4, 8), path)
    frame = pd.DataFrame(rows)
    split = {**parts, "train_sessions": [1], "dev_sessions": [2], "test_sessions": [3]}
    return frame, split


def _config(**training) -> dict:
    base = {
        "seed": 0,
        "batch_size": 2,
        "epochs": 1,
        "patience": 8,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "num_workers": 0,
        "device": "cpu",
        "loss_weights": {
            "emotion": 1.0,
            "va": 0.0,
            "alignment": 0.0,
            "orthogonality": 0.0,
            "audio_to_video_distillation": 0.0,
            "video_to_audio_distillation": 0.0,
        },
        "require_stage2_checkpoint": False,
        "allow_stage1_fallback": False,
        "max_nonfinite_batches": 1,
    }
    base.update(training)
    return {
        "data": {"has_va": True, "dataset_name": "synthetic"},
        "features": {
            "audio_max_tokens": 8,
            "video_max_frames": 4,
            "audio_dir": "audio",
            "video_dir": "video",
        },
        "model": {
            "hidden_dim": 16,
            "num_heads": 4,
            "fusion_layers": 1,
            "dropout": 0.0,
            "num_classes": 4,
        },
        "training": base,
    }


def _assert_metric_fields(payload: dict, *, status: str, model: str) -> None:
    for field in METRIC_FIELDS:
        assert field in payload
    assert payload["status"] == status
    assert payload["model"] == model
    assert payload["dataset_name"] == "synthetic"
    assert payload["fold"] == "fold_3"
    assert payload["seed"] == 0
    assert len(payload["config_sha256"]) == 64
    assert payload["git_commit"]
    assert payload["wall_clock_seconds"] >= 0


def test_stage2_failure_writes_metrics_and_no_checkpoint(tmp_path: Path) -> None:
    frame, split = _frame_and_features(tmp_path / "features")
    checkpoint = tmp_path / "essence_fold_3_blocked.pt"
    config = _config(
        require_stage2_checkpoint=True,
        allow_stage1_fallback=False,
        staged={
            "enabled": True,
            "stage1_epochs": 10,
            "stage1_loss_weights": {
                "emotion": 1.0,
                "va": 0.0,
                "alignment": 0.0,
                "orthogonality": 0.0,
                "audio_to_video_distillation": 0.0,
                "video_to_audio_distillation": 0.0,
            },
        },
    )
    with pytest.raises(Stage2NotReached):
        run_training(frame, split, tmp_path / "features", "essence", config, checkpoint)
    assert not checkpoint.exists()
    payload = json.loads(checkpoint.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    _assert_metric_fields(payload, status="failed", model="essence")
    assert payload["reason"] == "stage2_not_reached"
    assert payload["used_stage1_fallback"] is False
    assert payload["stage2_reached"] is False
    assert payload["aborted_nonfinite"] is False
    assert payload["run_id"] == checkpoint.stem


def test_existing_metrics_or_checkpoint_refuses_to_start(tmp_path: Path) -> None:
    frame, split = _frame_and_features(tmp_path / "features")
    checkpoint = tmp_path / "audio_fold_3_taken.pt"
    metrics = checkpoint.with_suffix(".metrics.json")
    metrics.write_text("{}\n", encoding="utf-8")
    with pytest.raises(CheckpointExistsError, match="new --tag"):
        run_training(frame, split, tmp_path / "features", "audio", _config(), checkpoint)
    assert metrics.read_text(encoding="utf-8") == "{}\n"
    assert not checkpoint.exists()

    metrics.unlink()
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(CheckpointExistsError):
        run_training(frame, split, tmp_path / "features", "audio", _config(), checkpoint)
    assert checkpoint.read_bytes() == b"checkpoint"


def test_complete_run_records_eligibility_fields(tmp_path: Path) -> None:
    frame, split = _frame_and_features(tmp_path / "features")
    checkpoint = tmp_path / "audio_fold_3_ok.pt"
    result = run_training(frame, split, tmp_path / "features", "audio", _config(), checkpoint)
    assert checkpoint.is_file()
    payload = json.loads(checkpoint.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    _assert_metric_fields(payload, status="complete", model="audio")
    assert payload["used_stage1_fallback"] is False
    assert payload["aborted_nonfinite"] is False
    assert result["status"] == "complete"
    assert not checkpoint.with_name(f".{checkpoint.name}.tmp").exists()


def test_evaluate_loader_marks_va_nan_without_labels(tmp_path: Path) -> None:
    class Stub(torch.nn.Module):
        def forward(self, audio, mask):
            batch = audio.shape[0]
            return {"logits": torch.zeros(batch, 4), "va": torch.ones(batch, 2)}

    frame = pd.DataFrame(
        {"utterance_id": ["crd_1"], "emotion4": ["sad"], "session": [1]}
    )
    dataset = MultimodalFeatureDataset(
        frame, ["crd_1"], tmp_path, load_audio=False, load_video=False, has_va=False
    )
    loader = DataLoader(dataset, batch_size=1, collate_fn=multimodal_collate)
    metrics, predictions = evaluate_loader(Stub(), "audio", loader, torch.device("cpu"))
    assert predictions["valence_target"].isna().all()
    assert predictions["activation_prediction"].isna().all()
    assert metrics["valence_ccc"] != metrics["valence_ccc"]
    assert metrics["va_mae"] != metrics["va_mae"]
