from __future__ import annotations
import csv
import gzip
import hashlib
import io
import shutil
import traceback
from pathlib import Path
import numpy as np
from .common import *

def keep_mask(length, fraction, uid, corruption_seed=20260926):
    if length < 1 or not 0 < fraction < 1:
        raise ValueError("Partial loss needs a nonempty sequence and fraction in (0,1)")
    digest = hashlib.sha256(f"{corruption_seed}:{uid}:video_tokens".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    removed = min(int(np.floor(length * fraction)), length - 1)
    keep = np.ones(length, dtype=bool)
    keep[rng.permutation(length)[:removed]] = False
    return keep

def alter_batch(batch, condition, corruption_seed):
    import torch
    out = dict(batch)
    if condition == "clean":
        pass
    elif condition in ("zero_audio", "zero_video"):
        modality = condition.removeprefix("zero_")
        out[modality] = torch.zeros_like(batch[modality])
    elif condition in ("branch_disabled_audio", "branch_disabled_video"):
        out["disable_" + condition.removeprefix("branch_disabled_")] = True
    elif condition.startswith("video_token_loss_"):
        fraction = int(condition.rsplit("_", 1)[-1]) / 100
        mask = batch["video_mask"].clone()
        for i, uid in enumerate(batch["utterance_id"]):
            positions = mask[i].nonzero(as_tuple=True)[0]
            keep = keep_mask(len(positions), fraction, uid, corruption_seed)
            mask[i, positions] = torch.as_tensor(keep, device=mask.device)
        out["video_mask"] = mask
    else:
        raise ValueError(f"Unsupported condition: {condition}")
    return out

def validate_export_selection(audit, models, max_runs, allow_unknown):
    selected = [r for r in audit["runs"] if not models or r["model"] in models]
    if models and set(models) - {r["model"] for r in selected}:
        raise ValueError("Unknown/absent model selection")
    if max_runs is not None:
        if max_runs < 1:
            raise ValueError("--max-runs must be positive")
        selected = selected[:max_runs]
    if not selected:
        raise ValueError("Empty run selection")
    if not audit["inputs_valid"]:
        raise ValueError("Audit has invalid features; resolve input failures first")
    rejected = [r["run_id"] for r in selected if r["status"] == "rejected"]
    if rejected:
        raise ValueError("Rejected checkpoints cannot be exported: " + ", ".join(rejected))
    unknown = [r["run_id"] for r in selected if r["status"] == "history_unknown"]
    if unknown and not allow_unknown:
        raise ValueError("Historical metadata is incomplete. Review audit.json. For explicitly flagged "
                         "re-evaluation only, use --include-history-unknown; this does not certify training.")
    return selected

def compare_reference_predictions(ids, labels, predictions, reference):
    required = {"utterance_id", "label", "prediction"}
    if not required.issubset(reference.columns) or reference.utterance_id.duplicated().any():
        raise ValueError("Invalid saved prediction reference")
    if set(reference.utterance_id) != set(ids) or len(reference) != len(ids):
        raise ValueError("Saved prediction reference has different IDs")
    aligned = reference.set_index("utterance_id").loc[ids]
    if not np.array_equal(aligned.label, labels):
        raise ValueError("Saved prediction labels differ from current labels")
    if not np.array_equal(aligned.prediction, predictions):
        changed = int(np.sum(aligned.prediction.to_numpy() != np.asarray(predictions)))
        raise ValueError(f"Clean predicted classes differ from the saved reference for {changed} utterances")

def export_one(repo, target, row, frame, allocation, feature_lookup, device, batch_size,
               corruption_seed, parity_tolerance):
    import torch
    from torch.utils.data import DataLoader
    from iemocap_mer.analysis import load_trained_model
    from iemocap_mer.training import forward_model, move_batch
    from iemocap_mer.data import multimodal_collate
    from .analysis import classification

    class StrictFeatures:
        def __init__(self, ids):
            self.rows = frame.set_index("utterance_id", drop=False).loc[ids].reset_index(drop=True)
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, index):
            data = self.rows.iloc[index]
            item = {"utterance_id": data.utterance_id, "label": LABELS.index(data.emotion4)}
            va = np.array([data.valence, data.activation], dtype=float)
            if not np.isfinite(va).all():
                raise ValueError("Non-finite IEMOCAP VA annotation")
            item["va"] = torch.tensor((np.clip(va, 1, 5) - 3) / 2, dtype=torch.float32)
            for modality, cache in (("audio", row["audio_dir"]), ("video", row["video_dir"])):
                if modality == "video" and row["model_name"] == "audio":
                    item[modality] = torch.zeros(1, 1)
                    continue
                entry = feature_lookup[(cache, data.utterance_id)]
                raw = within(repo, entry["relative_path"]).read_bytes()
                if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                    raise ValueError(f"Feature changed since audit: {cache}/{data.utterance_id}")
                value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
                if value.ndim != 2 or min(value.shape) < 1 or not bool(torch.isfinite(value).all()):
                    raise ValueError(f"Invalid feature: {cache}/{data.utterance_id}")
                value = value.float()
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("Float32 feature overflow")
                limit = row[modality + "_max_tokens"]
                if len(value) > limit:
                    indices = torch.linspace(0, len(value) - 1, limit).round().long()
                    value = value[indices]
                item[modality] = value
            return item

    checkpoint_path = within(repo, row["checkpoint_relative_path"])
    if sha256(checkpoint_path) != row["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed since audit")
    if sha256(checkpoint_path.with_suffix(".metrics.json")) != row["metrics_sha256"]:
        raise ValueError("Metrics changed since audit")
    model, checkpoint = load_trained_model(checkpoint_path, device)
    if row["model_name"] == "essence" and checkpoint.get("stage") != 2:
        raise ValueError("Refusing a non-Stage-2 Essence checkpoint")
    del checkpoint
    columns = ["dataset", "model", "seed", "fold", "run_id", "utterance_id", "condition", "split", "label",
        *[f"p_{i}" for i in range(4)], *[f"z_{i}" for i in range(4)],
        "video_valid_tokens", "video_kept_tokens", "corruption_seed"]
    conditions = applicable_conditions(row["model_name"])
    records, parity_ids, parity_labels, parity_predictions = {}, [], [], []
    model.eval()
    for part, name in (("dev", "validation"), ("test", "test")):
        path = target / f"{name}_predictions.csv.gz"
        loader = DataLoader(StrictFeatures(allocation[part]), batch_size=batch_size,
                            shuffle=False, num_workers=0, collate_fn=multimodal_collate)
        count = 0
        with gzip.open(path, "wt", encoding="utf-8", newline="") as handle, torch.inference_mode():
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for raw_batch in loader:
                batch = move_batch(raw_batch, device)
                for condition in (("clean",) if part == "dev" else conditions):
                    altered = alter_batch(batch, condition, corruption_seed)
                    z = forward_model(model, row["model_name"], altered)["logits"].detach().cpu().double().numpy()
                    if z.shape != (len(raw_batch["utterance_id"]), 4) or not np.isfinite(z).all():
                        raise ValueError("Expected finite Bx4 logits")
                    shifted = z - z.max(1, keepdims=True)
                    p = np.exp(shifted); p /= p.sum(1, keepdims=True)
                    labels = raw_batch["label"].numpy()
                    if part == "test" and condition == "clean":
                        parity_ids.extend(raw_batch["utterance_id"])
                        parity_labels.extend(labels.tolist()); parity_predictions.extend(z.argmax(1).tolist())
                    valid_counts = raw_batch["video_mask"].sum(-1).tolist()
                    kept_counts = altered["video_mask"].sum(-1).cpu().tolist()
                    for i, uid in enumerate(raw_batch["utterance_id"]):
                        record = dict(dataset="iemocap", model=row["model"], seed=row["seed"], fold=row["fold"],
                            run_id=row["run_id"], utterance_id=uid, condition=condition, split=name,
                            label=int(labels[i]), corruption_seed=corruption_seed,
                            video_valid_tokens=valid_counts[i] if row["model_name"] != "audio" else "",
                            video_kept_tokens=kept_counts[i] if row["model_name"] != "audio" else "")
                        record.update({f"p_{k}": float(p[i, k]) for k in range(4)})
                        record.update({f"z_{k}": float(z[i, k]) for k in range(4)})
                        writer.writerow(record); count += 1
        records[name] = dict(file=path.name, sha256=sha256(path), rows=count)
    clean_scores = classification(np.asarray(parity_labels), np.asarray(parity_predictions))
    difference = abs(clean_scores["macro_f1"] - float(row["reference_macro_f1"]))
    if difference > parity_tolerance:
        raise ValueError(f"Clean F1 mismatch: exported={clean_scores['macro_f1']}, "
                         f"stored={row['reference_macro_f1']}; inspect before using the results")
    parity_basis = "stored aggregate F1 only; per-utterance reference unavailable"
    if row.get("reference_predictions_relative_path"):
        import pandas as pd
        reference_path = within(repo, row["reference_predictions_relative_path"])
        if sha256(reference_path) != row["reference_predictions_sha256"]:
            raise ValueError("Saved reference predictions changed after audit")
        reference = pd.read_csv(reference_path, dtype={"utterance_id": str})
        compare_reference_predictions(parity_ids, parity_labels, parity_predictions, reference)
        parity_basis = "exact labels and predicted classes by utterance ID, plus aggregate F1"
    result = {"run_id": row["run_id"], "model": row["model"], "seed": row["seed"], "fold": row["fold"],
        "checkpoint_sha256": row["checkpoint_sha256"], "history_status": row["status"],
        "conditions": list(conditions), "files": records, "clean_scores": clean_scores,
        "reference_macro_f1": row["reference_macro_f1"], "parity_absolute_difference": difference,
        "clean_parity_basis": parity_basis,
        "created": timestamp()}
    write_json(target / "complete.json", result)
    return result

def run_export(args):
    import pandas as pd
    repo = repo_import(args.repo)
    from iemocap_mer.training import resolve_device
    audit_dir = Path(args.audit).resolve()
    audit = read_json(audit_dir / "audit.json")
    selected = validate_export_selection(audit, args.models, args.max_runs, args.include_history_unknown)
    for path, expected in audit["source_hashes"].items():
        if sha256(within(repo, path)) != expected:
            raise ValueError(f"Source/config changed since audit: {path}. Create a fresh audit.")
    for path, key in (("artifacts/manifests/iemocap_utterances.csv", "manifest_sha256"),
                      ("artifacts/splits/emotion4_session_folds.json", "splits_sha256")):
        if sha256(repo / path) != audit[key]:
            raise ValueError("Manifest/splits changed since audit")
    if sha256(audit_dir / "feature_manifest.csv") != audit["feature_manifest_sha256"]:
        raise ValueError("Feature inventory was edited after audit")
    features = pd.read_csv(audit_dir / "feature_manifest.csv", dtype=str)
    feature_lookup = {(r["cache"], r["utterance_id"]): r for r in features.to_dict("records")}
    frame = pd.read_csv(repo / "artifacts/manifests/iemocap_utterances.csv", dtype={"utterance_id": str})
    frame = frame[frame.emotion4.isin(LABELS)].copy()
    folds = read_json(repo / "artifacts/splits/emotion4_session_folds.json")
    validate_splits(frame, folds)
    device = resolve_device(args.device)
    settings = {"audit_sha256": sha256(audit_dir / "audit.json"), "selected": [r["run_id"] for r in selected],
        "split_manifest_sha256": sha256(audit_dir / "split_manifest.csv"), "toolkit_hashes": toolkit_hashes(),
        "device": str(device), "batch_size": args.batch_size, "corruption_seed": args.corruption_seed,
        "parity_tolerance": 1e-6, "include_history_unknown": args.include_history_unknown,
        "conditions": list(CONDITIONS)}
    out = Path(args.out).resolve()
    if args.resume:
        if read_json(out / "settings.json") != settings:
            raise ValueError("Resume settings differ; use a new output folder")
        if sha256(out / "split_manifest.csv") != settings["split_manifest_sha256"]:
            raise ValueError("Copied split manifest changed after export")
    else:
        out = new_output(out); write_json(out / "settings.json", settings)
        shutil.copyfile(audit_dir / "split_manifest.csv", out / "split_manifest.csv")
        shutil.copyfile(audit_dir / "run_manifest.csv", out / "run_manifest.csv")
    completed, failed = [], []
    for n, row in enumerate(selected, 1):
        target = out / "runs" / row["run_id"]
        if (target / "complete.json").exists():
            marker = read_json(target / "complete.json")
            for saved in marker["files"].values():
                if sha256(within(target, saved["file"])) != saved["sha256"]:
                    raise ValueError("Completed export was modified; do not resume over it")
            completed.append(marker)
            print(f"Reuse complete export {n}/{len(selected)}: {row['run_id']}", flush=True)
            continue
        if target.exists():
            archive = out / "failed_attempts" / (row["run_id"] + "_" + timestamp().replace(":", "-"))
            archive.parent.mkdir(parents=True, exist_ok=True); target.rename(archive)
        target.mkdir(parents=True, exist_ok=False)
        print(f"Export {n}/{len(selected)}: {row['run_id']}", flush=True)
        try:
            completed.append(export_one(repo, target, row, frame, folds[f"fold_{row['fold']}"],
                feature_lookup, device, args.batch_size, args.corruption_seed, 1e-6))
        except Exception as exc:
            failure = {"run_id": row["run_id"], "error": str(exc), "traceback": traceback.format_exc()}
            failed.append(failure); write_json(target / "failure.json", failure)
            print(f"FAILED {row['run_id']}: {exc}", flush=True)
        manifest = {"format_version": 1, "dataset": "iemocap", "settings": settings,
            "complete": len(completed) == len(selected) and not failed,
            "selected_count": len(selected), "expected_principal_count": 75,
            "history_unknown_count": sum(r["status"] == "history_unknown" for r in selected),
            "runs": completed, "failures": failed, "updated": timestamp(),
            "unimplemented": ["fully_masked_audio", "fully_masked_video"],
            "audio_only_applicability": "clean and zero_audio; video interventions are not applicable"}
        write_json(out / "export_manifest.json", manifest)
    # Also refresh the manifest if every run was reused on resume.
    manifest = {"format_version": 1, "dataset": "iemocap", "settings": settings,
        "complete": len(completed) == len(selected) and not failed,
        "selected_count": len(selected), "expected_principal_count": 75,
        "history_unknown_count": sum(r["status"] == "history_unknown" for r in selected),
        "runs": completed, "failures": failed, "updated": timestamp(),
        "unimplemented": ["fully_masked_audio", "fully_masked_video"],
        "audio_only_applicability": "clean and zero_audio; video interventions are not applicable"}
    write_json(out / "export_manifest.json", manifest)
    return 0 if manifest["complete"] else 2
