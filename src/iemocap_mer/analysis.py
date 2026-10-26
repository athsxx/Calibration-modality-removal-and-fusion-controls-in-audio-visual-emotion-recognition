from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from .data import MultimodalFeatureDataset, multimodal_collate
from .metrics import calculate_metrics
from .training import build_model, forward_model, move_batch, resolve_device


def load_trained_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(
        checkpoint["model_name"],
        checkpoint["audio_dim"],
        checkpoint["video_dim"],
        checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def _perturb(batch: dict, condition: str, noise_std: float = 0.5) -> dict:
    altered = dict(batch)
    if condition == "missing_audio":
        altered["audio"] = torch.zeros_like(batch["audio"])
    elif condition == "missing_video":
        altered["video"] = torch.zeros_like(batch["video"])
    elif condition == "noisy_audio":
        altered["audio"] = batch["audio"] + noise_std * torch.randn_like(batch["audio"])
    elif condition == "masked_video":
        altered["video"] = batch["video"].clone()
        altered["video"][:, ::2] = 0
    elif condition == "masked_audio_branch":
        altered["disable_audio"] = True
    elif condition == "masked_video_branch":
        altered["disable_video"] = True
    elif condition != "clean":
        raise ValueError(condition)
    return altered


@torch.inference_mode()
def robustness_evaluation(model, model_name: str, loader, device: torch.device) -> dict:
    conditions = (
        "clean",
        "missing_audio",
        "missing_video",
        "noisy_audio",
        "masked_video",
        "masked_audio_branch",
        "masked_video_branch",
    )
    result = {}
    for condition in conditions:
        labels, predictions, va_targets, va_predictions = [], [], [], []
        for raw_batch in loader:
            batch = _perturb(move_batch(raw_batch, device), condition)
            output = forward_model(model, model_name, batch)
            labels.append(batch["label"].cpu())
            predictions.append(output["logits"].argmax(-1).cpu())
            va_targets.append(batch["va"].cpu())
            va_predictions.append(output["va"].cpu())
        result[condition] = calculate_metrics(
            torch.cat(labels).numpy(),
            torch.cat(predictions).numpy(),
            torch.cat(va_targets).numpy(),
            torch.cat(va_predictions).numpy(),
        )
    clean_f1 = result["clean"]["macro_f1"]
    for condition in conditions[1:]:
        result[condition]["macro_f1_drop"] = clean_f1 - result[condition]["macro_f1"]
    return result


@torch.inference_mode()
def explanation_summary(model, model_name: str, loader, device: torch.device) -> pd.DataFrame:
    rows = []
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        clean = forward_model(model, model_name, batch)
        missing_audio = forward_model(model, model_name, _perturb(batch, "missing_audio"))
        missing_video = forward_model(model, model_name, _perturb(batch, "missing_video"))
        probabilities = clean["logits"].softmax(-1)
        missing_audio_probs = missing_audio["logits"].softmax(-1)
        missing_video_probs = missing_video["logits"].softmax(-1)
        chosen = probabilities.argmax(-1)
        ablated_audio_class = missing_audio_probs.argmax(-1)
        ablated_video_class = missing_video_probs.argmax(-1)
        clean_max_prob = probabilities.max(-1).values
        ablated_max_prob_without_audio = missing_audio_probs.max(-1).values
        ablated_max_prob_without_video = missing_video_probs.max(-1).values
        for index, utterance_id in enumerate(raw_batch["utterance_id"]):
            label = int(chosen[index])
            clean_confidence = float(clean_max_prob[index].cpu())
            ablated_audio_max = float(ablated_max_prob_without_audio[index].cpu())
            ablated_video_max = float(ablated_max_prob_without_video[index].cpu())
            row = {
                "utterance_id": utterance_id,
                "predicted_class": label,
                "confidence": clean_confidence,
                "clean_max_prob": clean_confidence,
                "ablated_max_prob_without_audio": ablated_audio_max,
                "ablated_max_prob_without_video": ablated_video_max,
                "ablated_predicted_class_audio": int(ablated_audio_class[index]),
                "ablated_predicted_class_video": int(ablated_video_class[index]),
                "exact_ci_audio": clean_confidence - ablated_audio_max,
                "exact_ci_video": clean_confidence - ablated_video_max,
                # Legacy columns: probability of the clean-predicted class under ablation.
                "confidence_drop_without_audio": float(
                    (probabilities[index, label] - missing_audio_probs[index, label]).cpu()
                ),
                "confidence_drop_without_video": float(
                    (probabilities[index, label] - missing_video_probs[index, label]).cpu()
                ),
            }
            if "reliability" in clean:
                row["audio_reliability"] = float(clean["reliability"][index, 0].cpu())
                row["video_reliability"] = float(clean["reliability"][index, 1].cpu())
            rows.append(row)
    return pd.DataFrame(rows)


def distress_proxy_analysis(predictions: pd.DataFrame, manifest: pd.DataFrame) -> dict:
    frame = predictions.merge(
        manifest[["utterance_id", "valence", "activation", "distress_proxy"]],
        on="utterance_id",
        how="inner",
    )
    target = frame.distress_proxy.astype(int).to_numpy()
    thresholds = (-0.25, 0.0, 0.25)
    rows = []
    for valence_threshold in thresholds:
        for activation_threshold in thresholds:
            predicted = (
                (frame.valence_prediction < valence_threshold)
                & (frame.activation_prediction > activation_threshold)
            ).astype(int)
            rows.append(
                {
                    "valence_threshold_normalized": valence_threshold,
                    "activation_threshold_normalized": activation_threshold,
                    "positive_predictions": int(predicted.sum()),
                    "precision": float(precision_score(target, predicted, zero_division=0)),
                    "recall": float(recall_score(target, predicted, zero_division=0)),
                    "f1": float(f1_score(target, predicted, zero_division=0)),
                }
            )
    return {
        "claim_boundary": "Exploratory VA-derived proxy; not a clinical distress endpoint.",
        "samples": int(len(frame)),
        "target_positives": int(target.sum()),
        "threshold_sensitivity": rows,
    }


def make_loader(
    frame,
    ids,
    feature_root,
    batch_size: int,
    model_name: str,
    *,
    audio_dir: str = "audio",
    video_dir: str = "video",
    video_max_tokens: int = 32,
    has_va: bool = True,
):
    return DataLoader(
        MultimodalFeatureDataset(
            frame,
            ids,
            feature_root,
            load_audio=model_name != "video",
            load_video=model_name != "audio",
            video_max_tokens=video_max_tokens,
            audio_dir=audio_dir,
            video_dir=video_dir,
            has_va=has_va,
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multimodal_collate,
    )


def run_posthoc(
    checkpoint_path: str | Path,
    frame: pd.DataFrame,
    split: dict,
    feature_root: str | Path,
    output_root: str | Path,
    device_name: str = "auto",
    batch_size: int = 16,
    *,
    audio_dir: str | None = None,
    video_dir: str | None = None,
    video_max_tokens: int = 32,
    has_va: bool = True,
) -> dict:
    device = resolve_device(device_name)
    model, checkpoint = load_trained_model(checkpoint_path, device)
    loader = make_loader(
        frame,
        split["test"],
        feature_root,
        batch_size,
        checkpoint["model_name"],
        audio_dir=audio_dir or checkpoint.get("audio_dir", "audio"),
        video_dir=video_dir or checkpoint.get("video_dir", "video"),
        video_max_tokens=video_max_tokens,
        has_va=has_va,
    )
    robustness = robustness_evaluation(model, checkpoint["model_name"], loader, device)
    explanations = explanation_summary(model, checkpoint["model_name"], loader, device)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stem = Path(checkpoint_path).stem
    explanations.to_csv(output_root / f"{stem}_explanations.csv", index=False)
    (output_root / f"{stem}_robustness.json").write_text(
        json.dumps(robustness, indent=2) + "\n", encoding="utf-8"
    )
    return robustness
