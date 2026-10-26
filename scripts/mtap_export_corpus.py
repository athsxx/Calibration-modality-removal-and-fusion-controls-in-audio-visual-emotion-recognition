#!/usr/bin/env python3
"""Corpus-generic identified logit export with the companion CSV contract.

Checkpoints supply architecture, feature directories and dimensions. Token
limits are read from the checkpoint when it stores them and otherwise fall
back to the IEMOCAP values (400 audio tokens, 32 visual tokens). Labels come
from emotion4 in the fixed order angry, happy_excited, neutral, sad. The
string ``happy`` is accepted as class 1 because protocol v3 uses that name on
RAVDESS and CREMA-D. Valence and activation are not read.

Video-only models export clean, zero_video and visual token-loss conditions.
Audio-only models export clean and zero_audio. Fusion models export the
companion condition list. Development exports are clean only.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "mer_companion"))

from iemocap_mer.analysis import load_trained_model
from iemocap_mer.data import multimodal_collate
from iemocap_mer.selective_fusion import PREDICTION_COLUMNS
from iemocap_mer.training import forward_model, move_batch, resolve_device
from merkit.analysis import classification, softmax
from merkit.common import (
    CONDITIONS,
    LABELS,
    iemocap_identity,
    new_output,
    read_json,
    sha256,
    timestamp,
    within,
    write_json,
)
from merkit.export import alter_batch, compare_reference_predictions

LABEL_INDEX = {name: index for index, name in enumerate(LABELS)}
LABEL_INDEX["happy"] = LABELS.index("happy_excited")
IEMOCAP_ID = re.compile(r"Ses\d{2}[FM]_.+_[FM]\d+")
CORRUPTION_SEED = 20260926


def parser() -> argparse.ArgumentParser:
    tool = argparse.ArgumentParser(description=__doc__)
    tool.add_argument("--artifacts-root", required=True)
    tool.add_argument("--manifest", required=True, help="Path relative to the artifacts root")
    tool.add_argument("--splits", required=True, help="Path relative to the artifacts root")
    tool.add_argument("--dataset", required=True)
    tool.add_argument("--run-spec", required=True, help="CSV with system, checkpoint, seed, fold")
    tool.add_argument("--out", required=True)
    tool.add_argument("--device", default="auto")
    tool.add_argument("--batch-size", type=int, default=16)
    tool.add_argument("--corruption-seed", type=int, default=CORRUPTION_SEED)
    tool.add_argument("--resume", action="store_true")
    tool.add_argument("--parity-tolerance", type=float, default=1e-6)
    return tool


def refuse_protected(path: Path) -> None:
    resolved = path.resolve()
    for parent in (resolved, *resolved.parents):
        if parent.name == "mtap_outputs" or parent.name.startswith("artifacts"):
            raise ValueError(f"Refusing to write under {parent}")


def applicable_conditions(model_name: str) -> tuple[str, ...]:
    if model_name == "audio":
        return ("clean", "zero_audio")
    if model_name == "video":
        return ("clean", "zero_video", "video_token_loss_25", "video_token_loss_50", "video_token_loss_75")
    return CONDITIONS


def assess_eligibility(metrics: dict, model_name: str, checkpoint: dict) -> tuple[str, list[str], list[str]]:
    """Return status, rejection reasons and unknown-history reasons.

    A missing metrics ``status`` is history_unknown. A present status other
    than ``complete`` is rejected. Essence runs with ``stage2_reached`` false
    or ``used_stage1_fallback`` true are rejected; missing Essence fields are
    history_unknown. A saved Essence checkpoint whose stage is not 2 is rejected.
    """
    rejected, unknown = [], []
    if "status" not in metrics:
        unknown.append("metrics status field absent")
    elif metrics.get("status") != "complete":
        rejected.append(f"metrics status is {metrics.get('status')!r}, not complete")
    if metrics.get("aborted_nonfinite") is True:
        rejected.append("run aborted for non-finite values")
    if model_name == "essence":
        if checkpoint.get("stage") != 2:
            rejected.append("saved Essence checkpoint is not Stage 2")
        if "stage2_reached" not in metrics:
            unknown.append("stage2_reached not recorded")
        elif metrics.get("stage2_reached") is not True:
            rejected.append("stage2_reached is not true")
        if "used_stage1_fallback" not in metrics:
            unknown.append("used_stage1_fallback not recorded")
        elif metrics.get("used_stage1_fallback"):
            rejected.append("used_stage1_fallback")
    if rejected:
        return "rejected", rejected, unknown
    if unknown:
        return "history_unknown", rejected, unknown
    return "eligible", rejected, unknown


def emotion_index(value) -> int:
    text = str(value).strip()
    if text not in LABEL_INDEX:
        raise ValueError(f"emotion4 {value!r} is outside {LABELS} (happy is accepted as class 1)")
    return LABEL_INDEX[text]


def token_limits(checkpoint: dict) -> tuple[int, int]:
    features = checkpoint.get("feature_config") or checkpoint.get("features") or {}
    if not isinstance(features, dict):
        features = {}
    audio = checkpoint.get("audio_max_tokens", features.get("audio_max_tokens", 400))
    video = checkpoint.get(
        "video_max_tokens",
        checkpoint.get("video_max_frames", features.get("video_max_frames", features.get("video_max_tokens", 32))),
    )
    audio, video = int(audio), int(video)
    if audio < 1 or video < 1:
        raise ValueError("Token limits must be positive")
    return audio, video


def locate_feature(feature_root: Path, cache: str, utterance_id: str, session) -> Path:
    name = f"{utterance_id}.pt"
    candidates = []
    if session is not None and str(session) != "":
        candidates.append(feature_root / cache / f"Session{int(session)}" / name)
    if IEMOCAP_ID.fullmatch(str(utterance_id)):
        parsed, _speaker = iemocap_identity(utterance_id)
        candidates.append(feature_root / cache / f"Session{parsed}" / name)
    candidates.append(feature_root / cache / name)
    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path
    raise FileNotFoundError(f"No feature file for {cache}/{utterance_id}")


def load_feature(path: Path, limit: int) -> torch.Tensor:
    raw = path.read_bytes()
    value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if value.ndim != 2 or min(value.shape) < 1 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"Invalid feature: {path}")
    value = value.float()
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"Float32 feature overflow: {path}")
    if len(value) > limit:
        indices = torch.linspace(0, len(value) - 1, limit).round().long()
        value = value[indices]
    return value


class StrictCorpusFeatures:
    """Feature dataset that does not read valence or activation."""

    def __init__(self, frame, ids, feature_root, row, audio_limit, video_limit):
        lookup = frame.set_index("utterance_id", drop=False)
        missing = [uid for uid in ids if uid not in lookup.index]
        if missing:
            raise ValueError(f"Split ids missing from the manifest: {missing[:5]}")
        self.rows = lookup.loc[list(ids)].reset_index(drop=True)
        if self.rows.utterance_id.duplicated().any():
            raise ValueError("Duplicate utterance ids in a split")
        self.feature_root = Path(feature_root)
        self.row = row
        self.audio_limit = audio_limit
        self.video_limit = video_limit

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        data = self.rows.iloc[index]
        item = {
            "utterance_id": str(data.utterance_id),
            "label": emotion_index(data.emotion4),
            "has_va": False,
            "va": torch.zeros(2, dtype=torch.float32),
        }
        session = data.session if "session" in data.index else None
        for modality, cache, limit in (
            ("audio", self.row["audio_dir"], self.audio_limit),
            ("video", self.row["video_dir"], self.video_limit),
        ):
            skip = (modality == "video" and self.row["model_name"] == "audio") or (
                modality == "audio" and self.row["model_name"] == "video"
            )
            if skip:
                item[modality] = torch.zeros(1, 1)
                continue
            path = locate_feature(self.feature_root, cache, item["utterance_id"], session)
            item[modality] = load_feature(path, limit)
        return item


def split_records(dataset: str, frame: pd.DataFrame, folds: dict) -> list[dict]:
    required = {"utterance_id", "emotion4", "session", "speaker_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if frame.utterance_id.astype(str).duplicated().any():
        raise ValueError("Duplicate utterance ids in the manifest")
    labelled = frame[frame.emotion4.map(lambda value: str(value).strip() in LABEL_INDEX)].copy()
    labelled["utterance_id"] = labelled.utterance_id.astype(str)
    lookup = labelled.set_index("utterance_id", drop=False)
    ids = set(lookup.index)
    records, seen_test = [], set()
    fold_names = sorted(folds, key=lambda name: int(str(name).split("_")[-1]))
    for fold_name in fold_names:
        fold = int(str(fold_name).split("_")[-1])
        allocation = folds[fold_name]
        groups = {}
        for part in ("train", "dev", "test"):
            values = [str(uid) for uid in allocation[part]]
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate split ids in fold {fold} {part}")
            unknown = [uid for uid in values if uid not in ids]
            if unknown:
                raise ValueError(f"Split ids are not labelled emotion4 utterances: {unknown[:5]}")
            groups[part] = values
        sets = {part: set(values) for part, values in groups.items()}
        names = list(sets)
        for left_i, left in enumerate(names):
            for right in names[left_i + 1:]:
                if sets[left] & sets[right]:
                    raise ValueError(f"Utterance leakage in fold {fold}")
                speakers_left = {str(lookup.loc[uid, "speaker_id"]) for uid in sets[left]}
                speakers_right = {str(lookup.loc[uid, "speaker_id"]) for uid in sets[right]}
                if speakers_left & speakers_right:
                    raise ValueError(f"Speaker leakage in fold {fold}")
        if seen_test & sets["test"]:
            raise ValueError("A test utterance occurs in more than one outer fold")
        seen_test |= sets["test"]
        for part, values in groups.items():
            split_name = "validation" if part == "dev" else part
            for uid in values:
                row = lookup.loc[uid]
                if IEMOCAP_ID.fullmatch(uid):
                    parsed_session, parsed_speaker = iemocap_identity(uid)
                    if int(row.session) != parsed_session or str(row.speaker_id) != parsed_speaker:
                        raise ValueError(f"Manifest identity disagrees with the utterance id {uid}")
                records.append({
                    "dataset": dataset,
                    "fold": fold,
                    "split": split_name,
                    "utterance_id": uid,
                    "session": int(row.session),
                    "speaker_id": str(row.speaker_id),
                    "label": emotion_index(row.emotion4),
                })
    return records


def write_split_manifest(path: Path, records: list[dict]) -> None:
    columns = ["dataset", "fold", "split", "utterance_id", "session", "speaker_id", "label"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def normalise_run_spec(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    renamed = {}
    for column in frame.columns:
        key = column.strip().lower()
        if key in {"system", "system_name", "model"}:
            renamed[column] = "system"
        elif key in {"checkpoint", "checkpoint_relative_path", "checkpoint_path"}:
            renamed[column] = "checkpoint"
        elif key == "seed":
            renamed[column] = "seed"
        elif key == "fold":
            renamed[column] = "fold"
    frame = frame.rename(columns=renamed)
    required = {"system", "checkpoint", "seed", "fold"}
    if not required.issubset(frame.columns):
        raise ValueError("Run spec needs system, checkpoint, seed and fold columns")
    if frame.empty:
        raise ValueError("Run spec is empty")
    frame["system"] = frame.system.astype(str)
    frame["checkpoint"] = frame.checkpoint.astype(str)
    frame["seed"] = frame.seed.astype(int)
    frame["fold"] = frame.fold.astype(int)
    if frame.duplicated(["system", "seed", "fold"]).any():
        raise ValueError("Duplicate system/seed/fold rows in the run spec")
    return frame.reset_index(drop=True)


def reference_macro_f1(metrics: dict):
    test = metrics.get("test")
    if isinstance(test, dict) and "macro_f1" in test:
        value = float(test["macro_f1"])
        if not np.isfinite(value):
            raise ValueError("Stored test macro-F1 is not finite")
        return value
    return None


def write_predictions(path: Path, rows: list[dict]) -> int:
    keys = [(row["utterance_id"], row["condition"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate utterance/condition rows in {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return len(rows)


def export_one(artifacts: Path, target: Path, spec: dict, frame, allocation, feature_root,
               device, batch_size, corruption_seed, parity_tolerance, dataset: str) -> dict:
    checkpoint_path = within(artifacts, spec["checkpoint"])
    metrics_path = checkpoint_path.with_suffix(".metrics.json")
    if not metrics_path.is_file():
        raise ValueError(f"Missing metrics file: {metrics_path.name}")
    metrics = read_json(metrics_path)
    model, checkpoint = load_trained_model(checkpoint_path, device)
    model_name = checkpoint.get("model_name")
    if not model_name:
        raise ValueError("Checkpoint has no model_name")
    status, rejected, unknown = assess_eligibility(metrics, model_name, checkpoint)
    if status == "rejected":
        raise ValueError("Rejected checkpoint: " + "; ".join(rejected))
    audio_limit, video_limit = token_limits(checkpoint)
    row = {
        "model_name": model_name,
        "audio_dir": checkpoint.get("audio_dir", "audio"),
        "video_dir": checkpoint.get("video_dir", "video"),
    }
    conditions = applicable_conditions(model_name)
    run_id = checkpoint_path.stem
    records = {}
    parity_ids, parity_labels, parity_predictions = [], [], []
    model.eval()
    for part, split_name in (("dev", "validation"), ("test", "test")):
        dataset_split = StrictCorpusFeatures(
            frame, allocation[part], feature_root, row, audio_limit, video_limit,
        )
        loader = DataLoader(
            dataset_split, batch_size=batch_size, shuffle=False, num_workers=0,
            collate_fn=multimodal_collate,
        )
        rows = []
        seen = set()
        with torch.inference_mode():
            for raw_batch in loader:
                ids = [str(uid) for uid in raw_batch["utterance_id"]]
                if any(uid in seen for uid in ids) or len(ids) != len(set(ids)):
                    raise ValueError("Duplicate utterance ids while exporting")
                seen.update(ids)
                batch = move_batch(raw_batch, device)
                labels = raw_batch["label"].detach().cpu().numpy()
                for condition in (("clean",) if part == "dev" else conditions):
                    altered = alter_batch(batch, condition, corruption_seed)
                    logits = forward_model(model, model_name, altered)["logits"].detach().cpu().double().numpy()
                    if logits.shape != (len(ids), 4) or not np.isfinite(logits).all():
                        raise ValueError("Expected finite Bx4 logits")
                    probabilities = softmax(logits)
                    if part == "test" and condition == "clean":
                        parity_ids.extend(ids)
                        parity_labels.extend(labels.tolist())
                        parity_predictions.extend(logits.argmax(1).tolist())
                    valid_counts = raw_batch["video_mask"].sum(-1).tolist()
                    kept_counts = altered["video_mask"].sum(-1).detach().cpu().tolist()
                    blank_video = model_name == "audio"
                    for index, uid in enumerate(ids):
                        record = {
                            "dataset": dataset,
                            "model": spec["system"],
                            "seed": int(spec["seed"]),
                            "fold": int(spec["fold"]),
                            "run_id": run_id,
                            "utterance_id": uid,
                            "condition": condition,
                            "split": split_name,
                            "label": int(labels[index]),
                            "video_valid_tokens": "" if blank_video else int(valid_counts[index]),
                            "video_kept_tokens": "" if blank_video else int(kept_counts[index]),
                            "corruption_seed": corruption_seed,
                        }
                        record.update({f"p_{k}": float(probabilities[index, k]) for k in range(4)})
                        record.update({f"z_{k}": float(logits[index, k]) for k in range(4)})
                        rows.append(record)
        count = write_predictions(target / f"{split_name}_predictions.csv.gz", rows)
        records[split_name] = {
            "file": f"{split_name}_predictions.csv.gz",
            "sha256": sha256(target / f"{split_name}_predictions.csv.gz"),
            "rows": count,
        }
    clean_scores = classification(np.asarray(parity_labels), np.asarray(parity_predictions))
    stored = reference_macro_f1(metrics)
    difference = None if stored is None else abs(clean_scores["macro_f1"] - stored)
    if difference is not None and difference > parity_tolerance:
        raise ValueError(
            f"Clean F1 mismatch: exported={clean_scores['macro_f1']}, stored={stored}"
        )
    parity_basis = "stored aggregate F1 only; per-utterance reference unavailable"
    reference_path = checkpoint_path.with_suffix(".predictions.csv")
    if reference_path.is_file():
        reference = pd.read_csv(reference_path, dtype={"utterance_id": str})
        compare_reference_predictions(parity_ids, parity_labels, parity_predictions, reference)
        parity_basis = "exact labels and predicted classes by utterance ID, plus aggregate F1"
        if stored is None:
            difference = 0.0
    elif stored is None:
        parity_basis = "no stored macro-F1 and no per-utterance reference"
        difference = None
    result = {
        "run_id": run_id,
        "model": spec["system"],
        "model_name": model_name,
        "seed": int(spec["seed"]),
        "fold": int(spec["fold"]),
        "checkpoint_sha256": sha256(checkpoint_path),
        "metrics_sha256": sha256(metrics_path),
        "history_status": status,
        "history_unknown_reasons": unknown,
        "conditions": list(conditions),
        "files": records,
        "clean_scores": clean_scores,
        "reference_macro_f1": stored,
        "parity_absolute_difference": difference,
        "clean_parity_basis": parity_basis,
        "audio_dir": row["audio_dir"],
        "video_dir": row["video_dir"],
        "audio_max_tokens": audio_limit,
        "video_max_tokens": video_limit,
        "created": timestamp(),
    }
    write_json(target / "complete.json", result)
    return result


def settings_payload(args, dataset, device, selected) -> dict:
    return {
        "dataset": dataset,
        "artifacts_root": str(Path(args.artifacts_root).resolve()),
        "manifest": args.manifest,
        "splits": args.splits,
        "selected": [row["run_id"] for row in selected],
        "device": str(device),
        "batch_size": args.batch_size,
        "corruption_seed": args.corruption_seed,
        "parity_tolerance": args.parity_tolerance,
        "conditions_audio": list(applicable_conditions("audio")),
        "conditions_video": list(applicable_conditions("video")),
        "conditions_fusion": list(applicable_conditions("concat_matched")),
    }


def refresh_manifest(out: Path, dataset: str, settings: dict, selected, completed, failed) -> dict:
    manifest = {
        "format_version": 1,
        "dataset": dataset,
        "settings": settings,
        "complete": len(completed) == len(selected) and not failed,
        "selected_count": len(selected),
        "expected_principal_count": len(selected),
        "history_unknown_count": sum(run.get("history_status") == "history_unknown" for run in completed),
        "runs": completed,
        "failures": failed,
        "updated": timestamp(),
        "audio_only_applicability": "clean and zero_audio",
        "video_only_applicability": "clean, zero_video and video_token_loss_25/50/75",
    }
    write_json(out / "export_manifest.json", manifest)
    return manifest


def export_corpus(args) -> int:
    artifacts = Path(args.artifacts_root).resolve()
    out = Path(args.out).resolve()
    refuse_protected(out)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    manifest_path = within(artifacts, args.manifest)
    splits_path = within(artifacts, args.splits)
    frame = pd.read_csv(manifest_path, dtype={"utterance_id": str})
    if "emotion4" not in frame.columns:
        raise ValueError("Manifest has no emotion4 column")
    folds = read_json(splits_path)
    records = split_records(args.dataset, frame, folds)
    spec = normalise_run_spec(Path(args.run_spec).resolve())
    selected = []
    for row in spec.to_dict("records"):
        checkpoint = within(artifacts, row["checkpoint"])
        selected.append({**row, "run_id": checkpoint.stem, "checkpoint": row["checkpoint"]})
    if len({row["run_id"] for row in selected}) != len(selected):
        raise ValueError("Duplicate checkpoint stems in the run spec")
    device = resolve_device(args.device)
    feature_root = artifacts / "features"
    settings = settings_payload(args, args.dataset, device, selected)
    if args.resume:
        if not out.is_dir():
            raise ValueError("Resume requires an existing output directory")
        previous = read_json(out / "settings.json")
        for key, value in settings.items():
            if previous.get(key) != value:
                raise ValueError(f"Resume settings differ on {key}; use a new output folder")
        if sha256(out / "split_manifest.csv") != previous.get("split_manifest_sha256"):
            raise ValueError("Split manifest changed after the export started")
        settings = previous
    else:
        out = new_output(out)
        write_split_manifest(out / "split_manifest.csv", records)
        settings["split_manifest_sha256"] = sha256(out / "split_manifest.csv")
        write_json(out / "settings.json", settings)
    completed, failed = [], []
    for number, row in enumerate(selected, 1):
        target = out / "runs" / row["run_id"]
        if (target / "complete.json").is_file():
            marker = read_json(target / "complete.json")
            for saved in marker["files"].values():
                if sha256(within(target, saved["file"])) != saved["sha256"]:
                    raise ValueError("Completed export was modified; do not resume over it")
            completed.append(marker)
            print(f"Reuse complete export {number}/{len(selected)}: {row['run_id']}", flush=True)
            continue
        if target.exists():
            archive = out / "failed_attempts" / (row["run_id"] + "_" + timestamp().replace(":", "-"))
            archive.parent.mkdir(parents=True, exist_ok=True)
            target.rename(archive)
        target.mkdir(parents=True, exist_ok=False)
        print(f"Export {number}/{len(selected)}: {row['run_id']}", flush=True)
        try:
            fold_key = f"fold_{int(row['fold'])}"
            if fold_key not in folds:
                raise ValueError(f"Splits have no {fold_key}")
            completed.append(export_one(
                artifacts, target, row, frame, folds[fold_key], feature_root, device,
                args.batch_size, args.corruption_seed, args.parity_tolerance, args.dataset,
            ))
        except Exception as exc:
            failure = {"run_id": row["run_id"], "error": str(exc), "traceback": traceback.format_exc()}
            failed.append(failure)
            write_json(target / "failure.json", failure)
            print(f"FAILED {row['run_id']}: {exc}", flush=True)
        refresh_manifest(out, args.dataset, settings, selected, completed, failed)
    manifest = refresh_manifest(out, args.dataset, settings, selected, completed, failed)
    print(f"Export {'complete' if manifest['complete'] else 'incomplete'}: {out}", flush=True)
    return 0 if manifest["complete"] else 2


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        return export_corpus(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
