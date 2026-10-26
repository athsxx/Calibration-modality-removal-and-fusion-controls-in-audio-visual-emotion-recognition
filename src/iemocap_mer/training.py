from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import MultimodalFeatureDataset, multimodal_collate
from .features import feature_path
from .losses import (
    cml_regularization_loss,
    combine_task_and_cml,
    faithful_cml_objective,
    faithful_training_loss,
    multitask_loss,
    predictive_confidence,
    weighted_cross_entropy,
)
from .metrics import calculate_metrics
from .models import (
    ConcatFusionModel,
    ConcatFusionModelMatched,
    CrossAttentionBaseline,
    EssenceTransferModel,
    UnimodalModel,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_modality_dim(
    frame: pd.DataFrame,
    feature_root: str | Path,
    modality: str,
    *,
    audio_dir: str = "audio",
    video_dir: str = "video",
) -> int:
    feature_dir = video_dir if modality == "video" else audio_dir if modality == "audio" else None
    sessions = frame["session"].tolist() if "session" in frame.columns else [None] * len(frame)
    for utterance_id, session in zip(frame.utterance_id, sessions):
        session_value = None if pd.isna(session) else int(session)
        path = feature_path(
            feature_root, modality, utterance_id, feature_dir=feature_dir, session=session_value
        )
        if path.exists():
            value = torch.load(path, map_location="cpu", weights_only=True)
            return int(value.shape[-1])
    raise FileNotFoundError(f"No {modality} feature files were found")


def build_model(name: str, audio_dim: int, video_dim: int, config: dict) -> torch.nn.Module:
    common = {
        "hidden_dim": int(config["hidden_dim"]),
        "num_classes": int(config["num_classes"]),
        "dropout": float(config["dropout"]),
    }
    if name == "audio":
        return UnimodalModel(audio_dim, temporal_stride=4, **common)
    if name == "video":
        return UnimodalModel(video_dim, **common)
    if name == "concat":
        return ConcatFusionModel(audio_dim, video_dim, **common)
    if name == "concat_matched":
        extra = {}
        if config.get("fusion_width") is not None:
            extra["fusion_width"] = int(config["fusion_width"])
        if config.get("encoder_layers") is not None:
            extra["encoder_layers"] = int(config["encoder_layers"])
        return ConcatFusionModelMatched(audio_dim, video_dim, **common, **extra)
    if name == "cross_attention":
        return CrossAttentionBaseline(audio_dim, video_dim, **common)
    if name == "essence":
        return EssenceTransferModel(
            audio_dim,
            video_dim,
            num_heads=int(config["num_heads"]),
            fusion_layers=int(config["fusion_layers"]),
            **common,
        )
    raise ValueError(f"Unknown model: {name}")


def forward_model(model, name: str, batch: dict) -> dict:
    if name == "audio":
        return model(batch["audio"], batch["audio_mask"])
    if name == "video":
        return model(batch["video"], batch["video_mask"])
    kwargs = {}
    if name in ("concat_matched", "essence"):
        if batch.get("disable_audio"):
            kwargs["disable_audio"] = True
        if batch.get("disable_video"):
            kwargs["disable_video"] = True
    return model(batch["audio"], batch["audio_mask"], batch["video"], batch["video_mask"], **kwargs)


def batch_with_zeroed_modality(batch: dict, modality: str) -> dict:
    """Zero one branch's features, leaving its mask unchanged.

    This is the train-time form of the export ``zero_audio`` / ``zero_video``
    removal (feature values replaced by zeros; the mask is not edited and the
    branch is not disabled).
    """
    if modality not in ("audio", "video"):
        raise ValueError(f"Unknown modality: {modality}")
    return {**batch, modality: torch.zeros_like(batch[modality])}


def cml_penalty_for_batch(model, model_name: str, batch: dict, full_logits: torch.Tensor) -> torch.Tensor:
    """Confidence hinge for audio removed and for video removed, summed."""
    conf_full = predictive_confidence(full_logits)
    removed = []
    for modality in ("audio", "video"):
        altered = batch_with_zeroed_modality(batch, modality)
        removed_logits = forward_model(model, model_name, altered)["logits"]
        removed.append(predictive_confidence(removed_logits))
    return cml_regularization_loss(conf_full, *removed)


def faithful_cml_for_batch(
    model,
    model_name: str,
    batch: dict,
    full_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    modality: str,
    emotion_weight: float,
    cml_weight: float,
    ce_full: torch.Tensor | None = None,
) -> torch.Tensor:
    """Algorithm 1 for two modalities: one zero-filled partial input.

    ``modality`` is the modality erased on this step. Erasure is
    ``batch_with_zeroed_modality`` (feature values zeroed, mask unchanged),
    the same removal the export calls ``zero_audio`` / ``zero_video``.
    """
    if modality not in ("audio", "video"):
        raise ValueError(f"Unknown modality: {modality}")
    if ce_full is None:
        ce_full = weighted_cross_entropy(full_logits, labels, emotion_weight)
    altered = batch_with_zeroed_modality(batch, modality)
    partial_logits = forward_model(model, model_name, altered)["logits"]
    ce_partial = weighted_cross_entropy(partial_logits, labels, emotion_weight)
    return faithful_cml_objective(
        ce_full,
        [ce_partial],
        predictive_confidence(full_logits),
        [predictive_confidence(partial_logits)],
        cml_weight=cml_weight,
    )


def apply_modality_dropout(batch: dict, p_drop_audio: float, p_drop_video: float) -> dict:
    """Train-time per-sample modality dropout (zero-fill, matching missing_* semantics)."""
    if p_drop_audio <= 0.0 and p_drop_video <= 0.0:
        return batch
    batch_size = batch["audio"].shape[0]
    if p_drop_audio > 0.0:
        drop = torch.rand(batch_size, device=batch["audio"].device) < p_drop_audio
        if drop.any():
            audio = batch["audio"].clone()
            audio[drop] = 0
            batch = {**batch, "audio": audio}
    if p_drop_video > 0.0:
        drop = torch.rand(batch_size, device=batch["video"].device) < p_drop_video
        if drop.any():
            video = batch["video"].clone()
            video[drop] = 0
            batch = {**batch, "video": video}
    return batch


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.inference_mode()
def evaluate_loader(model, model_name: str, loader, device: torch.device) -> tuple[dict, pd.DataFrame]:
    model.eval()
    rows: list[dict] = []
    labels, predictions, va_targets, va_predictions = [], [], [], []
    has_va = True
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        batch_has_va = bool(batch.get("has_va", True))
        has_va = has_va and batch_has_va
        output = forward_model(model, model_name, batch)
        predicted = output["logits"].argmax(dim=-1)
        labels.append(batch["label"].cpu())
        predictions.append(predicted.cpu())
        if batch_has_va:
            va_targets.append(batch["va"].cpu())
            va_predictions.append(output["va"].cpu())
        for index, utterance_id in enumerate(raw_batch["utterance_id"]):
            if batch_has_va:
                valence_target = float(batch["va"][index, 0].cpu())
                activation_target = float(batch["va"][index, 1].cpu())
                valence_prediction = float(output["va"][index, 0].cpu())
                activation_prediction = float(output["va"][index, 1].cpu())
            else:
                valence_target = activation_target = float("nan")
                valence_prediction = activation_prediction = float("nan")
            row = {
                "utterance_id": utterance_id,
                "label": int(batch["label"][index].cpu()),
                "prediction": int(predicted[index].cpu()),
                "valence_target": valence_target,
                "activation_target": activation_target,
                "valence_prediction": valence_prediction,
                "activation_prediction": activation_prediction,
            }
            if "reliability" in output:
                row["audio_reliability"] = float(output["reliability"][index, 0].cpu())
                row["video_reliability"] = float(output["reliability"][index, 1].cpu())
            rows.append(row)
    labels_np = torch.cat(labels).numpy()
    predictions_np = torch.cat(predictions).numpy()
    if has_va:
        targets_np = torch.cat(va_targets).numpy()
        va_np = torch.cat(va_predictions).numpy()
    else:
        targets_np = np.zeros((len(labels_np), 2), dtype=np.float64)
        va_np = np.zeros((len(labels_np), 2), dtype=np.float64)
    return (
        calculate_metrics(labels_np, predictions_np, targets_np, va_np, include_va=has_va),
        pd.DataFrame(rows),
    )


def active_loss_weights(training_config: dict, model_name: str, epoch: int) -> dict:
    staged = training_config.get("staged")
    if model_name != "essence" or not staged or not staged.get("enabled"):
        weights = dict(training_config["loss_weights"])
    elif epoch <= int(staged.get("stage1_epochs", 10)):
        weights = dict(staged.get("stage1_loss_weights", training_config["loss_weights"]))
    else:
        weights = dict(training_config["loss_weights"])
    # Default keeps the yaml distillation weights. Opt-in false zeros both
    # directional terms in every stage, matching --disable-loss on both names.
    if not bool(training_config.get("directional_distillation", True)):
        weights["audio_to_video_distillation"] = 0.0
        weights["video_to_audio_distillation"] = 0.0
    return weights


def training_stage(training_config: dict, model_name: str, epoch: int) -> int:
    staged = training_config.get("staged")
    if model_name != "essence" or not staged or not staged.get("enabled"):
        return 1
    return 1 if epoch <= int(staged.get("stage1_epochs", 10)) else 2


class NonFiniteTrainingAbort(RuntimeError):
    """Raised when too many non-finite loss/gradient batches occur.

    A numerically compromised run is a failure, not a Stage-1 fallback success.
    """

    def __init__(self, message: str, *, count: int, events: list[dict]):
        super().__init__(message)
        self.count = count
        self.events = events


class Stage2NotReached(RuntimeError):
    """Essence run required Stage 2 and never saved that checkpoint."""


class CheckpointExistsError(RuntimeError):
    """Target checkpoint or metrics file already exists. A new tag is required."""


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unknown"
    commit = completed.stdout.strip()
    if completed.returncode != 0 or not commit:
        return "unknown"
    return commit


def _config_sha256(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _fold_label(split: dict) -> str:
    sessions = split.get("test_sessions") or []
    if len(sessions) == 1:
        return f"fold_{int(sessions[0])}"
    return ""


def _eligibility_fields(
    *,
    config: dict,
    checkpoint_path: Path,
    model_name: str,
    split: dict,
    started: float,
    status: str,
    nonfinite_batch_count: int,
    aborted_nonfinite: bool,
    stage2_reached: bool,
    used_stage1_fallback: bool,
    reason: str | None = None,
) -> dict:
    fields = {
        "run_id": checkpoint_path.stem,
        "dataset_name": str(config.get("data", {}).get("dataset_name", "iemocap")),
        "status": status,
        "nonfinite_batch_count": int(nonfinite_batch_count),
        "aborted_nonfinite": bool(aborted_nonfinite),
        "stage2_reached": bool(stage2_reached),
        "used_stage1_fallback": bool(used_stage1_fallback),
        "config_sha256": _config_sha256(config),
        "git_commit": _git_commit(),
        "wall_clock_seconds": float(time.perf_counter() - started),
        "seed": int(config["training"]["seed"]),
        "fold": _fold_label(split),
        "model": model_name,
    }
    if reason is not None:
        fields["reason"] = reason
    return fields


def classify_tensor_nonfinite(value: torch.Tensor | None) -> tuple[bool, bool]:
    """Return (has_nan, has_inf) for a tensor; both False if value is None."""
    if value is None:
        return False, False
    return bool(torch.isnan(value).any().item()), bool(torch.isinf(value).any().item())


def classify_grad_nonfinite(model: torch.nn.Module) -> tuple[bool, bool]:
    """Return (has_nan, has_inf) across parameter gradients."""
    has_nan = False
    has_inf = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        nan, inf = classify_tensor_nonfinite(parameter.grad)
        has_nan = has_nan or nan
        has_inf = has_inf or inf
        if has_nan and has_inf:
            break
    return has_nan, has_inf


class NonFiniteMonitor:
    """Log non-finite loss/grad batches and abort after a threshold."""

    def __init__(self, event_path: Path, max_batches: int = 1):
        self.event_path = Path(event_path)
        self.max_batches = int(max_batches)
        self.count = 0
        self.events: list[dict] = []
        if self.event_path.exists():
            self.event_path.unlink()

    def record(
        self,
        *,
        epoch: int,
        batch_index: int,
        stage: int,
        loss_nan: bool,
        loss_inf: bool,
        grad_nan: bool,
        grad_inf: bool,
        where: str,
    ) -> None:
        self.count += 1
        event = {
            "epoch": epoch,
            "batch_index": batch_index,
            "stage": stage,
            "loss_nan": bool(loss_nan),
            "loss_inf": bool(loss_inf),
            "grad_nan": bool(grad_nan),
            "grad_inf": bool(grad_inf),
            "where": where,
            "count": self.count,
        }
        self.events.append(event)
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        with self.event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        print(json.dumps({"nonfinite_batch": event}))
        if self.count >= self.max_batches:
            raise NonFiniteTrainingAbort(
                f"Aborting after {self.count} non-finite batch(es) "
                f"(threshold={self.max_batches}) at epoch {epoch} batch {batch_index} "
                f"({where})",
                count=self.count,
                events=self.events,
            )

    def check_loss(self, loss: torch.Tensor, *, epoch: int, batch_index: int, stage: int) -> bool:
        """Return True if loss is finite. Record and possibly abort otherwise."""
        loss_nan, loss_inf = classify_tensor_nonfinite(loss)
        if not (loss_nan or loss_inf):
            return True
        self.record(
            epoch=epoch,
            batch_index=batch_index,
            stage=stage,
            loss_nan=loss_nan,
            loss_inf=loss_inf,
            grad_nan=False,
            grad_inf=False,
            where="loss",
        )
        return False

    def check_grads(self, model: torch.nn.Module, *, epoch: int, batch_index: int, stage: int) -> bool:
        """Return True if grads are finite. Record and possibly abort otherwise."""
        grad_nan, grad_inf = classify_grad_nonfinite(model)
        if not (grad_nan or grad_inf):
            return True
        self.record(
            epoch=epoch,
            batch_index=batch_index,
            stage=stage,
            loss_nan=False,
            loss_inf=False,
            grad_nan=grad_nan,
            grad_inf=grad_inf,
            where="grad",
        )
        return False


def _write_nonfinite_failure_metrics(
    *,
    checkpoint_path: Path,
    history: list[dict],
    require_stage2: bool,
    stage2_reached: bool,
    seed: int,
    abort: NonFiniteTrainingAbort,
    config: dict,
    model_name: str,
    split: dict,
    started: float,
) -> dict:
    """Persist a failed-run metrics file. Never marks Stage-1 fallback as success."""
    result = {
        "best_epoch": None,
        "dev": None,
        "test": None,
        "history": history,
        "require_stage2_checkpoint": require_stage2,
        "nonfinite_events": abort.events,
        "error": str(abort),
        **_eligibility_fields(
            config=config,
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            split=split,
            started=started,
            status="failed",
            nonfinite_batch_count=abort.count,
            aborted_nonfinite=True,
            stage2_reached=stage2_reached,
            used_stage1_fallback=False,
            reason="nonfinite_abort",
        ),
    }
    result["seed"] = seed
    _atomic_write_text(
        checkpoint_path.with_suffix(".metrics.json"),
        json.dumps(result, indent=2) + "\n",
    )
    print(json.dumps({"error": "nonfinite_abort", "count": abort.count, "message": str(abort)}))
    return result


def run_training(
    frame: pd.DataFrame,
    split: dict,
    feature_root: str | Path,
    model_name: str,
    config: dict,
    checkpoint_path: str | Path,
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    metrics_path = checkpoint_path.with_suffix(".metrics.json")
    if checkpoint_path.exists() or metrics_path.exists():
        raise CheckpointExistsError(
            f"Refusing to overwrite existing run '{checkpoint_path.stem}'. "
            f"checkpoint_exists={checkpoint_path.exists()} "
            f"metrics_exists={metrics_path.exists()}. Choose a new --tag."
        )
    started = time.perf_counter()
    training_config = config["training"]
    set_seed(int(training_config["seed"]))
    device_name = training_config["device"]
    if model_name == "video" and training_config.get("video_device"):
        device_name = training_config["video_device"]
    device = resolve_device(device_name)
    video_dir = str(config["features"].get("video_dir", "video"))
    audio_dir = str(config["features"].get("audio_dir", "audio"))
    has_va = bool(config.get("data", {}).get("has_va", True))
    audio_dim = (
        infer_modality_dim(frame, feature_root, "audio", audio_dir=audio_dir)
        if model_name != "video"
        else 1
    )
    video_dim = (
        infer_modality_dim(frame, feature_root, "video", video_dir=video_dir)
        if model_name != "audio"
        else 1
    )
    model = build_model(model_name, audio_dim, video_dim, config["model"]).to(device)

    loaders = {}
    for part in ("train", "dev", "test"):
        dataset = MultimodalFeatureDataset(
            frame,
            split[part],
            feature_root,
            load_audio=model_name != "video",
            load_video=model_name != "audio",
            audio_max_tokens=int(config["features"]["audio_max_tokens"]),
            video_max_tokens=int(config["features"]["video_max_frames"]),
            audio_dir=audio_dir,
            video_dir=video_dir,
            has_va=has_va,
        )
        loaders[part] = DataLoader(
            dataset,
            batch_size=int(training_config["batch_size"]),
            shuffle=part == "train",
            num_workers=int(training_config["num_workers"]),
            collate_fn=multimodal_collate,
        )
    learning_rate = float(training_config["learning_rate"])
    if model_name == "essence" and training_config.get("essence_learning_rate") is not None:
        learning_rate = float(training_config["essence_learning_rate"])
    patience = int(training_config["patience"])
    if model_name == "essence" and training_config.get("essence_patience") is not None:
        patience = int(training_config["essence_patience"])
    modality_dropout = training_config.get("modality_dropout", {})
    p_drop_audio = float(modality_dropout.get("audio", 0.0))
    p_drop_video = float(modality_dropout.get("video", 0.0))
    cml_weight = float(training_config.get("cml_weight", 0.0))
    cml_variant = str(training_config.get("cml_variant", "penalty"))
    if cml_variant not in ("penalty", "faithful"):
        raise ValueError("training.cml_variant must be 'penalty' or 'faithful'")
    if cml_weight < 0.0:
        raise ValueError("training.cml_weight must be >= 0")
    if cml_weight > 0.0 and model_name in ("audio", "video"):
        raise ValueError("CML regularization requires a fusion model")
    # Independent of the torch RNG so modality sampling does not change dropout
    # or data-order randomness. Unused when the variant is the existing penalty.
    cml_sampler = random.Random(int(training_config["seed"])) if cml_variant == "faithful" else None

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(training_config["weight_decay"]),
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    allow_stage1_fallback = bool(training_config.get("allow_stage1_fallback", False))
    require_stage2 = model_name == "essence" and bool(
        training_config.get("require_stage2_checkpoint", False)
    )
    best_f1, stale, history = -1.0, 0, []
    best_gate_f1 = -1.0
    last_stage = 0
    # In-memory best weights for the case where Stage 2 is never reached
    # without a non-finite abort (e.g. early stop still in Stage 1).
    fallback_checkpoint: dict | None = None
    stage2_reached = False
    max_nonfinite_batches = int(training_config.get("max_nonfinite_batches", 1))
    nonfinite = NonFiniteMonitor(
        checkpoint_path.with_suffix(".nonfinite_events.jsonl"),
        max_batches=max_nonfinite_batches,
    )

    def _snapshot(epoch: int, stage: int, dev_metrics: dict) -> dict:
        return {
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "model_name": model_name,
            "audio_dim": audio_dim,
            "video_dim": video_dim,
            "audio_dir": audio_dir,
            "video_dir": video_dir,
            "model_config": config["model"],
            "epoch": epoch,
            "stage": stage,
            "dev_metrics": dev_metrics,
        }

    for epoch in range(1, int(training_config["epochs"]) + 1):
        stage = training_stage(training_config, model_name, epoch)
        if stage != last_stage:
            stale = 0
            last_stage = stage
        if stage == 2:
            stage2_reached = True
        loss_weights = active_loss_weights(training_config, model_name, epoch)
        model.train()
        total_loss, batches = 0.0, 0
        try:
            for batch_index, raw_batch in enumerate(loaders["train"]):
                batch = move_batch(raw_batch, device)
                batch = apply_modality_dropout(batch, p_drop_audio, p_drop_video)
                optimizer.zero_grad(set_to_none=True)
                output = forward_model(model, model_name, batch)
                losses = multitask_loss(
                    output,
                    batch["label"],
                    batch["va"],
                    loss_weights,
                    has_va=bool(batch.get("has_va", True)),
                )
                loss = losses["total"]
                if cml_weight > 0.0:
                    # Extra forwards run only when the weight is positive, so a
                    # zero weight does not change dropout randomness or the loss.
                    if cml_variant == "penalty":
                        penalty = cml_penalty_for_batch(model, model_name, batch, output["logits"])
                        loss = combine_task_and_cml(loss, cml_weight, penalty)
                    else:
                        if cml_sampler is None:
                            raise RuntimeError("faithful CML requires a modality sampler")
                        emotion_weight = float(loss_weights.get("emotion", 0.0))
                        modality = cml_sampler.choice(("audio", "video"))
                        faithful_term = faithful_cml_for_batch(
                            model,
                            model_name,
                            batch,
                            output["logits"],
                            batch["label"],
                            modality=modality,
                            emotion_weight=emotion_weight,
                            cml_weight=cml_weight,
                            ce_full=emotion_weight * losses["emotion"],
                        )
                        loss = faithful_training_loss(
                            losses["total"], losses["emotion"], emotion_weight, faithful_term
                        )
                if not nonfinite.check_loss(loss, epoch=epoch, batch_index=batch_index, stage=stage):
                    continue
                loss.backward()
                if not nonfinite.check_grads(model, epoch=epoch, batch_index=batch_index, stage=stage):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                grad_clip = float(training_config.get("grad_clip", 1.0))
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                batches += 1
        except NonFiniteTrainingAbort as abort:
            _write_nonfinite_failure_metrics(
                checkpoint_path=checkpoint_path,
                history=history,
                require_stage2=require_stage2,
                stage2_reached=stage2_reached,
                seed=int(training_config["seed"]),
                abort=abort,
                config=config,
                model_name=model_name,
                split=split,
                started=started,
            )
            raise
        if batches == 0:
            print(json.dumps({"epoch": epoch, "error": "all training batches produced non-finite loss"}))
            break
        dev_metrics, _ = evaluate_loader(model, model_name, loaders["dev"], device)
        epoch_row = {
            "epoch": epoch,
            "stage": stage,
            "train_loss": total_loss / max(1, batches),
            **{f"dev_{key}": value for key, value in dev_metrics.items() if isinstance(value, float)},
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row))

        # `best_f1`/`stale` drive early stopping using the best macro-F1 seen so
        # far at *any* stage. `best_gate_f1` drives which checkpoint is actually
        # saved to disk under the Stage-2 gate. Independently, we always keep an
        # in-memory fallback of the absolute best epoch so a collapse before
        # Stage 2 still yields a usable (Stage-1) checkpoint rather than crashing.
        if dev_metrics["macro_f1"] > best_f1:
            best_f1, stale = dev_metrics["macro_f1"], 0
            fallback_checkpoint = _snapshot(epoch, stage, dev_metrics)
        else:
            stale += 1

        eligible_for_checkpoint = stage == 2 if require_stage2 else True
        if eligible_for_checkpoint and dev_metrics["macro_f1"] > best_gate_f1:
            best_gate_f1 = dev_metrics["macro_f1"]
            _atomic_torch_save(_snapshot(epoch, stage, dev_metrics), checkpoint_path)

        if stale >= patience:
            if require_stage2 and stage < 2:
                # Guarantee the run actually reaches Stage 2 before early
                # stopping is allowed to fire, so a Stage-2 checkpoint exists
                # whenever training remains numerically stable that long.
                stale = 0
            else:
                break

    used_stage1_fallback = False
    if not checkpoint_path.exists():
        if require_stage2 and not allow_stage1_fallback:
            result = {
                "best_epoch": None,
                "dev": None,
                "test": None,
                "history": history,
                "require_stage2_checkpoint": require_stage2,
                "error": "Stage 2 was not reached and Stage-1 fallback is disabled.",
                **_eligibility_fields(
                    config=config,
                    checkpoint_path=checkpoint_path,
                    model_name=model_name,
                    split=split,
                    started=started,
                    status="failed",
                    nonfinite_batch_count=nonfinite.count,
                    aborted_nonfinite=False,
                    stage2_reached=stage2_reached,
                    used_stage1_fallback=False,
                    reason="stage2_not_reached",
                ),
            }
            _atomic_write_text(metrics_path, json.dumps(result, indent=2) + "\n")
            raise Stage2NotReached(
                f"Stage 2 was not reached for {checkpoint_path.name}; "
                "no Stage-1 fallback checkpoint was saved. Choose a new --tag to retry."
            )
        if fallback_checkpoint is None:
            raise RuntimeError(
                f"No checkpoint was saved for {checkpoint_path.name}: training "
                "produced no finite epoch before any weights could be retained."
            )
        # Opt-in only: persist the best Stage-1 weights when Stage 2 never
        # produced a checkpoint. Default runs fail instead of saving this file.
        used_stage1_fallback = True
        _atomic_torch_save(fallback_checkpoint, checkpoint_path)
        print(
            json.dumps(
                {
                    "warning": "require_stage2_checkpoint set but Stage 2 never "
                    "produced a checkpoint; falling back to best Stage-1 weights",
                    "fallback_epoch": fallback_checkpoint["epoch"],
                    "stage2_reached": stage2_reached,
                    "nonfinite_batch_count": nonfinite.count,
                }
            )
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics, predictions = evaluate_loader(model, model_name, loaders["test"], device)
    predictions.to_csv(checkpoint_path.with_suffix(".predictions.csv"), index=False)
    result = {
        "best_epoch": checkpoint["epoch"],
        "dev": checkpoint["dev_metrics"],
        "test": test_metrics,
        "history": history,
        "require_stage2_checkpoint": require_stage2,
        "cml_weight": cml_weight,
        "cml_variant": cml_variant,
        **_eligibility_fields(
            config=config,
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            split=split,
            started=started,
            status="complete",
            nonfinite_batch_count=nonfinite.count,
            aborted_nonfinite=False,
            stage2_reached=stage2_reached,
            used_stage1_fallback=used_stage1_fallback,
        ),
    }
    _atomic_write_text(metrics_path, json.dumps(result, indent=2) + "\n")
    return result
