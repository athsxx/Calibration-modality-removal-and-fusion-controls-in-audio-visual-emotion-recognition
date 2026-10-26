from __future__ import annotations
import io
import hashlib
from pathlib import Path
from .common import *

def run_audit(args):
    import torch
    import pandas as pd
    import yaml
    repo = repo_import(args.repo)
    out = new_output(args.out)
    write_json(out / "status.json", {"status": "running", "started": timestamp()})
    try:
        manifest = repo / "artifacts/manifests/iemocap_utterances.csv"
        split_path = repo / "artifacts/splits/emotion4_session_folds.json"
        frame = pd.read_csv(manifest, dtype={"utterance_id": str})
        frame = frame[frame.emotion4.isin(LABELS)].copy()
        if len(frame) != 5531:
            raise ValueError(f"Expected 5531 IEMOCAP emotion4 rows; found {len(frame)}")
        splits = read_json(split_path)
        split_rows = validate_splits(frame, splits)
        write_csv(out / "split_manifest.csv", split_rows)
        source_hashes = {str(p.relative_to(repo)): sha256(p)
            for folder in ("src", "configs") for p in sorted((repo / folder).rglob("*"))
            if p.is_file() and p.suffix in (".py", ".yaml", ".yml")}
        features, failures, dimensions = [], [], {}
        for cache in ("audio_w2v", "video_v2", "video_au"):
            dimensions[cache] = set()
            for n, uid in enumerate(sorted(frame.utterance_id), 1):
                relative = feature_relative(cache, uid)
                row = dict(cache=cache, utterance_id=uid, relative_path=relative)
                try:
                    raw = within(repo, relative).read_bytes()
                    row["sha256"] = hashlib.sha256(raw).hexdigest()
                    tensor = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
                    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or min(tensor.shape) < 1:
                        raise ValueError("feature must be a nonempty two-dimensional tensor")
                    if not bool(torch.isfinite(tensor).all()) or not bool(torch.isfinite(tensor.float()).all()):
                        raise ValueError("feature contains non-finite values before/after float32 conversion")
                    row.update(tokens=int(tensor.shape[0]), dimension=int(tensor.shape[1]), status="valid")
                    dimensions[cache].add(row["dimension"])
                except Exception as exc:
                    row.update(status="invalid", error=type(exc).__name__ + ": " + str(exc).replace(str(repo), "<repo>"))
                    failures.append(row.copy())
                features.append(row)
                if n % 1000 == 0:
                    print(f"Feature audit: {cache} {n}/{len(frame)}", flush=True)
        for cache, dims in dimensions.items():
            if len(dims) != 1:
                failures.append({"cache": cache, "error": f"inconsistent feature dimensions: {sorted(dims)}"})
        write_csv(out / "feature_manifest.csv", features)
        feature_hash = sha256(out / "feature_manifest.csv")
        write_json(out / "feature_failures.json", failures)
        runs = []
        for spec in expected_runs():
            row = dict(spec, feature_manifest_sha256=feature_hash,
                training_commit="unknown (not stored in checkpoint)",
                historical_feature_sanitized_count="unknown",
                config_basis="current repository configuration; historical training snapshot not proven")
            checkpoint_path = within(repo, spec["checkpoint_relative_path"])
            metrics_path = checkpoint_path.with_suffix(".metrics.json")
            try:
                config_path = within(repo, spec["config_relative_path"])
                config = yaml.safe_load(config_path.read_text())
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                metrics = read_json(metrics_path)
                status, rejected, unknown = metadata_eligibility(spec, checkpoint, metrics)
                row.update(status=status, rejected_reasons=rejected, unknown_history=unknown,
                    checkpoint_sha256=sha256(checkpoint_path), metrics_sha256=sha256(metrics_path),
                    config_sha256=sha256(config_path), audio_max_tokens=int(config["features"]["audio_max_tokens"]),
                    video_max_tokens=int(config["features"]["video_max_frames"]),
                    checkpoint_stage=checkpoint.get("stage"), best_epoch=checkpoint.get("epoch"),
                    stage2_reached=metrics.get("stage2_reached", "unknown"),
                    used_stage1_fallback=metrics.get("used_stage1_fallback", "unknown"),
                    nonfinite_batch_count=metrics.get("nonfinite_batch_count", "unknown"),
                    reference_macro_f1=metrics.get("test", {}).get("macro_f1"),
                    reference_uar=metrics.get("test", {}).get("uar"),
                    audio_dim=checkpoint.get("audio_dim"), video_dim=checkpoint.get("video_dim"))
                if checkpoint.get("model_config") != config["model"]:
                    row["rejected_reasons"].append("current model configuration differs from checkpoint")
                for modality, cache in (("audio", "audio_w2v"), ("video", spec["video_dir"])):
                    if modality == "video" and spec["model_name"] == "audio":
                        continue
                    if dimensions.get(cache) != {checkpoint.get(modality + "_dim")}:
                        row["rejected_reasons"].append(f"{modality} feature dimension mismatch")
                if row["rejected_reasons"]:
                    row["status"] = "rejected"
                prediction_path = checkpoint_path.with_suffix(".predictions.csv")
                if prediction_path.is_file():
                    row["reference_predictions_relative_path"] = str(prediction_path.relative_to(repo))
                    row["reference_predictions_sha256"] = sha256(prediction_path)
                else:
                    row["unknown_history"].append("stored per-utterance prediction reference is missing")
                    if row["status"] == "eligible":
                        row["status"] = "history_unknown"
                del checkpoint
            except Exception as exc:
                row.update(status="rejected", rejected_reasons=[type(exc).__name__ + ": " +
                    str(exc).replace(str(repo), "<repo>")], unknown_history=[])
            runs.append(row)
            print(f"{row['status']}: {spec['run_id']}", flush=True)
        payload = {"format_version": 1, "baseline_commit": BASE_COMMIT, "created": timestamp(),
            "repository": git_info(repo), "source_hashes": source_hashes,
            "manifest_sha256": sha256(manifest), "splits_sha256": sha256(split_path),
            "feature_manifest_sha256": feature_hash, "inputs_valid": not failures,
            "counts": {s: sum(r["status"] == s for r in runs)
                       for s in ("eligible", "history_unknown", "rejected")},
            "runs": runs,
            "notes": ["No training occurred.", "Unknown historical fields are not treated as zero.",
                "Current feature integrity does not prove original training feature integrity.",
                "Audit enforces saved stage; it does not modify the existing trainer."]}
        write_json(out / "audit.json", payload)
        write_csv(out / "run_manifest.csv", runs)
        write_json(out / "status.json", {"status": "complete", "inputs_valid": not failures,
            "counts": payload["counts"], "finished": timestamp()})
        print(f"Audit saved: {out}", flush=True)
        return 0 if not failures and payload["counts"]["rejected"] == 0 else 2
    except Exception as exc:
        write_json(out / "status.json", {"status": "failed", "error": str(exc), "finished": timestamp()})
        raise
