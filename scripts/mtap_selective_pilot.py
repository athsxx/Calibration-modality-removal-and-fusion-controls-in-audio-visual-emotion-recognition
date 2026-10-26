#!/usr/bin/env python3
"""Protocol-v3 selective fusion on an identified logit export.

The anchor is chosen per outer fold from seed-averaged development macro-F1.
The gate, temperatures and tau are fitted per seed and fold on clean
development rows only. Test labels are written into the export so later
scoring can use them; they are not read while the gate is fit or applied.

When no video-only AU system is in the export, the anchor is the available
unimodal system and weighted logit averaging / QMF are skipped.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "mer_companion"))

from iemocap_mer.robust_stats import (  # noqa: E402
    BOOT_SEED,
    _speaker_axis,
    accumulate_effects,
    accumulate_speakers,
    bootstrap_component_verdict,
    consensus_verdict,
    effect_rates,
    fold_component_verdict,
    metrics_from_speaker_stats,
    paired_t_test,
    percentile_intervals,
    speaker_multiplicities,
    utterance_parts,
)
from iemocap_mer.selective_fusion import (  # noqa: E402
    FUSED_TO_SELECTIVE,
    LATE_FUSION_SKIPPED_NOTE,
    LOGIT_WEIGHT_GRID,
    PREDICTION_COLUMNS,
    TAU_GRID,
    VIDEO_ABSENT_CONDITIONS,
    VISUAL_TOKEN_BUDGET,
    averaged_logits,
    choose_anchor,
    fit_gate,
    gate_feature_matrix,
    is_audio_unimodal,
    is_unimodal_anchor_candidate,
    is_video_au_unimodal,
    qmf_fusion,
    select_logit_weight,
    select_tau,
    selective_logits,
    visual_evidence,
)
from merkit.analysis import classification, fit_temperature, holm, softmax  # noqa: E402
from merkit.common import PRIMARY_PAIRS, SEEDS, read_json, sha256, timestamp, write_json  # noqa: E402
from merkit.export import keep_mask  # noqa: E402

LOGIT_COLUMNS = [f"z_{k}" for k in range(4)]
VIDEO_DIRECTORIES = {
    "concatmatched_au": "video_au",
    "concat_matched_au": "video_au",
    "essence_au": "video_au",
    "video_au": "video_au",
    "iemocap_video_au": "video_au",
    "video_only_au": "video_au",
}
# Export manifests use concatmatched_au on IEMOCAP and concat_matched_au on the
# later corpora. Both are the protocol ConcatMatched-AU system. Original files
# keep the name they were exported with.
FUSED_EXPORT_ALIASES = {
    "concat_matched_au": "concatmatched_au",
}


def parser() -> argparse.ArgumentParser:
    tool = argparse.ArgumentParser(description=__doc__)
    tool.add_argument("--exports", required=True)
    tool.add_argument("--out", required=True)
    tool.add_argument("--boot", type=int, default=2000)
    tool.add_argument("--hboot", type=int, default=2000)
    tool.add_argument("--feature-root", help="Read-only visual features if token columns are absent")
    return tool


def refuse_protected(path: Path) -> None:
    resolved = path.resolve()
    for parent in (resolved, *resolved.parents):
        if parent.name == "mtap_outputs" or parent.name.startswith("artifacts"):
            raise ValueError(f"Refusing to write under {parent}")


def protocol_fused_name(model: str) -> str | None:
    """Map an export model name onto a protocol fused-system name, if it is one."""
    if model in FUSED_TO_SELECTIVE:
        return model
    alias = FUSED_EXPORT_ALIASES.get(model)
    if alias in FUSED_TO_SELECTIVE:
        return alias
    return None


def load_needed(exports: Path, manifest: dict) -> dict:
    loaded = {}
    for run in manifest["runs"]:
        model = run["model"]
        if protocol_fused_name(model) is None and not is_unimodal_anchor_candidate(model):
            continue
        frames = {}
        for part in ("validation", "test"):
            record = run["files"][part]
            path = exports / "runs" / run["run_id"] / record["file"]
            if sha256(path) != record["sha256"]:
                raise ValueError(f"Prediction file changed since export: {path}")
            frame = pd.read_csv(path, dtype={"utterance_id": str})
            if len(frame) != int(record["rows"]):
                raise ValueError(f"Row count changed for {path}")
            frames[part] = frame
        loaded[(model, int(run["seed"]), int(run["fold"]))] = {"run": run, "frames": frames}
    return loaded


def condition_frame(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
    group = frame[frame.condition.astype(str) == condition]
    if group.empty:
        raise ValueError(f"Missing condition {condition}")
    if group.utterance_id.duplicated().any():
        raise ValueError(f"Duplicate utterance ids in condition {condition}")
    return group.sort_values("utterance_id", kind="mergesort").reset_index(drop=True)


def align_to(reference: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    if set(reference.utterance_id) != set(other.utterance_id):
        raise ValueError("Anchor and fused utterance ids differ")
    ordered = other.set_index("utterance_id").loc[reference.utterance_id.to_numpy()].reset_index()
    if not np.array_equal(reference.label.to_numpy(), ordered.label.to_numpy()):
        raise ValueError("Anchor and fused labels differ")
    return ordered


def logits_of(frame: pd.DataFrame) -> np.ndarray:
    values = frame[LOGIT_COLUMNS].to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError("Expected finite Nx4 logits")
    return values


def evidence_for(frame: pd.DataFrame, condition: str, sessions: dict, feature_root: Path | None,
                 video_dir: str | None) -> np.ndarray:
    ids = frame.utterance_id.astype(str).to_numpy()
    if condition in VIDEO_ABSENT_CONDITIONS:
        return np.zeros(len(frame), dtype=np.float64)
    if "video_kept_tokens" in frame.columns:
        kept = pd.to_numeric(frame["video_kept_tokens"], errors="coerce")
        if bool(kept.notna().all()):
            return visual_evidence(condition, kept.to_numpy(dtype=np.float64))
    valid = None
    if "video_valid_tokens" in frame.columns:
        parsed = pd.to_numeric(frame["video_valid_tokens"], errors="coerce")
        if bool(parsed.notna().all()):
            valid = parsed.to_numpy(dtype=np.float64)
    if valid is None:
        if feature_root is None or not video_dir:
            raise ValueError(f"No visual-token columns for {condition} and no feature root was provided")
        lengths = []
        for uid in ids:
            path = feature_file(feature_root, video_dir, uid, sessions[uid])
            value = __import__("torch").load(path, map_location="cpu", weights_only=True)
            length = int(value.shape[0])
            lengths.append(float(min(length, VISUAL_TOKEN_BUDGET)))
        valid = np.asarray(lengths, dtype=np.float64)
    if condition.startswith("video_token_loss_"):
        fraction = int(condition.rsplit("_", 1)[-1]) / 100.0
        kept = [float(keep_mask(int(length), fraction, uid).sum()) for uid, length in zip(ids, valid)]
        return visual_evidence(condition, kept)
    return visual_evidence(condition, valid)


def feature_file(feature_root: Path, video_dir: str, utterance_id: str, session) -> Path:
    name = f"{utterance_id}.pt"
    candidates = [
        feature_root / video_dir / f"Session{int(session)}" / name,
        feature_root / video_dir / name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"No visual feature for {video_dir}/{utterance_id}")


def anchor_condition(available: set[str], condition: str, anchor_is_audio: bool) -> str:
    if condition in available:
        return condition
    video_intervention = condition in VIDEO_ABSENT_CONDITIONS or condition.startswith("video_token_loss_")
    audio_intervention = condition in {"zero_audio", "branch_disabled_audio"}
    if anchor_is_audio and video_intervention and "clean" in available:
        return "clean"
    if (not anchor_is_audio) and audio_intervention and "clean" in available:
        return "clean"
    raise ValueError(f"Anchor has no logits for condition {condition}")


def write_table(path: Path, rows: list[dict]) -> dict:
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
    return {"file": path.name, "sha256": sha256(path), "rows": len(rows)}


def prediction_rows(dataset, model, seed, fold, run_id, split, condition, ids, labels, logits,
                    source: pd.DataFrame) -> list[dict]:
    probabilities = softmax(logits)
    rows = []
    corruption = source["corruption_seed"].tolist() if "corruption_seed" in source.columns else [""] * len(ids)
    valid = source["video_valid_tokens"].tolist() if "video_valid_tokens" in source.columns else [""] * len(ids)
    kept = source["video_kept_tokens"].tolist() if "video_kept_tokens" in source.columns else [""] * len(ids)
    for index, uid in enumerate(ids):
        record = {
            "dataset": dataset,
            "model": model,
            "seed": int(seed),
            "fold": int(fold),
            "run_id": run_id,
            "utterance_id": uid,
            "condition": condition,
            "split": split,
            "label": int(labels[index]),
            "video_valid_tokens": valid[index],
            "video_kept_tokens": kept[index],
            "corruption_seed": corruption[index],
        }
        record.update({f"p_{k}": float(probabilities[index, k]) for k in range(4)})
        record.update({f"z_{k}": float(logits[index, k]) for k in range(4)})
        rows.append(record)
    return rows


def macro_of(labels, logits) -> float:
    return float(classification(np.asarray(labels, dtype=int), softmax(logits).argmax(1))["macro_f1"])


def blank(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    return value


def fit_dev_temperature(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    clean = condition_frame(frame, "clean")
    labels = clean.label.to_numpy(dtype=int)
    logits = logits_of(clean)
    temperature, _info = fit_temperature(labels, logits)
    return clean.utterance_id.astype(str).to_numpy(), labels, logits, float(temperature)


def selective_pilot(args) -> dict:
    exports = Path(args.exports).resolve()
    out = Path(args.out).resolve()
    refuse_protected(out)
    if out.exists():
        raise ValueError(f"Refusing to overwrite {out}")
    if args.boot < 1 or args.hboot < 1:
        raise ValueError("Bootstrap counts must be positive")
    manifest = read_json(exports / "export_manifest.json")
    if not manifest.get("complete", False):
        raise ValueError("Source export is not complete")
    loaded = load_needed(exports, manifest)
    present = {model for model, _seed, _fold in loaded}
    fused_pairs = {}
    for protocol_name, selective_name in FUSED_TO_SELECTIVE.items():
        found = [name for name in present if protocol_fused_name(name) == protocol_name]
        if len(found) != 1:
            raise ValueError(f"Expected one export of {protocol_name}, found {found}")
        fused_pairs[found[0]] = selective_name
    cmau_name = next(name for name, selective in fused_pairs.items() if selective == "selective_cmau")
    esau_name = next(name for name, selective in fused_pairs.items() if selective == "selective_esau")
    unimodal = sorted(name for name in present if is_unimodal_anchor_candidate(name))
    if not unimodal:
        raise ValueError("No unimodal system is available for the anchor")
    audio_models = [name for name in unimodal if is_audio_unimodal(name)]
    video_models = [name for name in unimodal if is_video_au_unimodal(name)]
    if len(audio_models) > 1 or len(video_models) > 1:
        raise ValueError(f"Expected at most one audio and one video-AU system, found {audio_models} / {video_models}")
    late_fusion_note = "" if audio_models and video_models else LATE_FUSION_SKIPPED_NOTE
    folds = sorted({fold for _model, _seed, fold in loaded if _model == cmau_name})
    seeds = tuple(sorted({seed for _model, seed, _fold in loaded if _model == cmau_name}))
    if tuple(seeds) != tuple(SEEDS) or tuple(folds) != (1, 2, 3, 4, 5):
        raise ValueError(f"Selective pilot expects seeds {SEEDS} and folds 1-5, found {seeds} / {folds}")

    split = pd.read_csv(exports / "split_manifest.csv", dtype={"utterance_id": str})
    sessions = {
        str(row.utterance_id): row.session
        for row in split[split["split"] == "test"].itertuples(index=False)
    }
    dev_sessions = {
        str(row.utterance_id): row.session
        for row in split[split["split"] == "validation"].itertuples(index=False)
    }
    sessions.update(dev_sessions)
    feature_root = Path(args.feature_root).resolve() if args.feature_root else REPO / "artifacts" / "features"
    dataset = manifest.get("dataset", "iemocap")
    anchors = {}
    anchor_scores = {}
    for fold in folds:
        scores = {}
        for model in unimodal:
            per_seed = []
            for seed in seeds:
                dev = loaded[(model, seed, fold)]["frames"]["validation"]
                _ids, labels, logits, _temperature = fit_dev_temperature(dev)
                per_seed.append(macro_of(labels, logits))
            scores[model] = float(np.mean(per_seed))
        anchors[fold] = choose_anchor(scores)
        anchor_scores[fold] = scores

    selective_root = out / "selective_export"
    selective_root.mkdir(parents=True)
    shutil.copyfile(exports / "split_manifest.csv", selective_root / "split_manifest.csv")
    store = {}
    diagnostics = []
    completed = []
    temperatures = {}

    def remember(model, seed, fold, condition, ids, labels, logits):
        store[(model, int(seed), int(fold), condition)] = (
            np.asarray(ids).astype(str),
            np.asarray(labels, dtype=int),
            np.asarray(logits, dtype=np.float64),
        )

    for model, seed, fold in list(loaded):
        dev_ids, dev_y, dev_z, temperature = fit_dev_temperature(loaded[(model, seed, fold)]["frames"]["validation"])
        temperatures[(model, seed, fold)] = temperature
        remember(model, seed, fold, "clean", dev_ids, dev_y, dev_z / temperature)
        test = loaded[(model, seed, fold)]["frames"]["test"]
        for condition in sorted(set(test.condition.astype(str))):
            group = condition_frame(test, condition)
            remember(model, seed, fold, condition, group.utterance_id, group.label, logits_of(group) / temperature)

    for fold in folds:
        anchor_name = anchors[fold]
        anchor_is_audio = is_audio_unimodal(anchor_name)
        for seed in seeds:
            print(f"Selective fold {fold} seed {seed} anchor {anchor_name}", flush=True)
            anchor_test = loaded[(anchor_name, seed, fold)]["frames"]["test"]
            anchor_conditions = set(anchor_test.condition.astype(str))
            dev_anchor = condition_frame(loaded[(anchor_name, seed, fold)]["frames"]["validation"], "clean")
            anchor_temperature = temperatures[(anchor_name, seed, fold)]
            weight, weight_f1, qmf_f1 = None, None, None
            if not late_fusion_note:
                audio_name, video_name = audio_models[0], video_models[0]
                audio_dev = condition_frame(loaded[(audio_name, seed, fold)]["frames"]["validation"], "clean")
                video_dev = align_to(audio_dev, condition_frame(
                    loaded[(video_name, seed, fold)]["frames"]["validation"], "clean"
                ))
                weight, weight_f1 = select_logit_weight(
                    audio_dev.label.to_numpy(dtype=int),
                    logits_of(audio_dev), temperatures[(audio_name, seed, fold)],
                    logits_of(video_dev), temperatures[(video_name, seed, fold)],
                )
                qmf_z, _weights = qmf_fusion(
                    logits_of(audio_dev), temperatures[(audio_name, seed, fold)],
                    logits_of(video_dev), temperatures[(video_name, seed, fold)],
                )
                qmf_f1 = macro_of(audio_dev.label, qmf_z)
            for fused_name, selective_name in fused_pairs.items():
                fused_dev = align_to(dev_anchor, condition_frame(
                    loaded[(fused_name, seed, fold)]["frames"]["validation"], "clean"
                ))
                fused_temperature = temperatures[(fused_name, seed, fold)]
                anchor_dev_z = logits_of(dev_anchor) / anchor_temperature
                fused_dev_z = logits_of(fused_dev) / fused_temperature
                anchor_dev_p = softmax(anchor_dev_z)
                fused_dev_p = softmax(fused_dev_z)
                dev_visual = evidence_for(
                    fused_dev, "clean", sessions, feature_root, VIDEO_DIRECTORIES.get(fused_name),
                )
                dev_features = gate_feature_matrix(anchor_dev_p, fused_dev_p, dev_visual)
                labels = dev_anchor.label.to_numpy(dtype=int)
                gate, gate_note, n_fit = fit_gate(
                    dev_features, labels, anchor_dev_p.argmax(1), fused_dev_p.argmax(1),
                )
                dev_gate = gate.probability(dev_features)
                tau, selective_dev_f1 = select_tau(dev_gate, anchor_dev_p, fused_dev_p, labels)
                _dev_logits, dev_switched = selective_logits(dev_gate, tau, anchor_dev_z, fused_dev_z)
                run_id = f"{selective_name}_fold_{fold}_seed{seed}"
                dev_rows = prediction_rows(
                    dataset, selective_name, seed, fold, run_id, "validation", "clean",
                    dev_anchor.utterance_id.astype(str), labels, _dev_logits, fused_dev,
                )
                test_rows = []
                clean_coverage = None
                written_conditions = []
                skipped_conditions = []
                fused_test = loaded[(fused_name, seed, fold)]["frames"]["test"]
                for condition in sorted(set(fused_test.condition.astype(str))):
                    try:
                        used = anchor_condition(anchor_conditions, condition, anchor_is_audio)
                    except ValueError:
                        # branch_disabled_audio is not a video intervention and is
                        # absent from audio-only exports, so no anchor logit is invented.
                        skipped_conditions.append(condition)
                        continue
                    fused_rows = condition_frame(fused_test, condition)
                    anchor_rows = align_to(fused_rows, condition_frame(anchor_test, used))
                    anchor_z = logits_of(anchor_rows) / anchor_temperature
                    fused_z = logits_of(fused_rows) / fused_temperature
                    evidence = evidence_for(
                        fused_rows, condition, sessions, feature_root, VIDEO_DIRECTORIES.get(fused_name),
                    )
                    features = gate_feature_matrix(softmax(anchor_z), softmax(fused_z), evidence)
                    chosen, switched = selective_logits(gate.probability(features), tau, anchor_z, fused_z)
                    ids = fused_rows.utterance_id.astype(str).to_numpy()
                    y = fused_rows.label.to_numpy(dtype=int)
                    remember(selective_name, seed, fold, condition, ids, y, chosen)
                    test_rows.extend(prediction_rows(
                        dataset, selective_name, seed, fold, run_id, "test", condition,
                        ids, y, chosen, fused_rows,
                    ))
                    if condition == "clean":
                        clean_coverage = float(switched.mean())
                    written_conditions.append(condition)
                if "clean" not in written_conditions:
                    raise ValueError(f"Selective run {run_id} produced no clean test rows")
                target = selective_root / "runs" / run_id
                target.mkdir(parents=True)
                files = {
                    "validation": write_table(target / "validation_predictions.csv.gz", dev_rows),
                    "test": write_table(target / "test_predictions.csv.gz", test_rows),
                }
                record = {
                    "run_id": run_id,
                    "model": selective_name,
                    "seed": int(seed),
                    "fold": int(fold),
                    "history_status": "derived",
                    "conditions": written_conditions,
                    "skipped_conditions": skipped_conditions,
                    "files": files,
                    "anchor": anchor_name,
                    "fused_system": fused_name,
                    "tau": tau,
                    "anchor_temperature": anchor_temperature,
                    "fused_temperature": fused_temperature,
                    "created": timestamp(),
                }
                write_json(target / "complete.json", record)
                completed.append(record)
                tied = [name for name, score in anchor_scores[fold].items() if score == max(anchor_scores[fold].values())]
                diagnostics.append({
                    "fold": fold,
                    "seed": seed,
                    "anchor": anchor_name,
                    "anchor_tie": len(tied) > 1,
                    "anchor_dev_macro_f1_seed_mean": anchor_scores[fold][anchor_name],
                    "selective_system": selective_name,
                    "fused_system": fused_name,
                    "tau": tau,
                    "coverage_dev": float(dev_switched.mean()),
                    "coverage_test": clean_coverage,
                    "dev_macro_f1_anchor": macro_of(labels, anchor_dev_z),
                    "dev_macro_f1_fused": macro_of(labels, fused_dev_z),
                    "dev_macro_f1_selective": selective_dev_f1,
                    "dev_macro_f1_logit_avg": blank(weight_f1),
                    "dev_macro_f1_qmf": blank(qmf_f1),
                    "logit_avg_audio_weight": blank(weight),
                    "n_fit": n_fit,
                    "gate_note": gate_note,
                    "late_fusion_note": late_fusion_note,
                })
            if not late_fusion_note:
                write_late_fusion(
                    dataset, seed, fold, audio_models[0], video_models[0], weight, loaded, temperatures,
                    sessions, selective_root, completed, store,
                )

    if late_fusion_note:
        (selective_root / "late_fusion_baselines.txt").write_text(late_fusion_note + "\n", encoding="utf-8")
    diagnostic_frame = pd.DataFrame(diagnostics)
    diagnostic_frame.to_csv(selective_root / "gate_diagnostics.csv", index=False)
    settings = {
        "dataset": dataset,
        "source_exports": str(exports),
        "source_export_manifest_sha256": sha256(exports / "export_manifest.json"),
        "split_manifest_sha256": sha256(selective_root / "split_manifest.csv"),
        "tau_grid": list(TAU_GRID),
        "logit_weight_grid": list(LOGIT_WEIGHT_GRID),
        "visual_token_budget": VISUAL_TOKEN_BUDGET,
        "late_fusion": late_fusion_note or "computed on development data",
        "test_labels_used_for_fitting": False,
    }
    selective_manifest = {
        "format_version": 1,
        "dataset": dataset,
        "settings": settings,
        "complete": True,
        "selected_count": len(completed),
        "expected_principal_count": len(completed),
        "history_unknown_count": 0,
        "runs": completed,
        "failures": [],
        "updated": timestamp(),
        "late_fusion_baselines": late_fusion_note or "logit_avg_av and qmf_av",
    }
    anchor_names = [anchors[fold] for fold in folds]
    if len(set(anchor_names)) == 1:
        anchor_model = anchor_names[0]
    else:
        anchor_model = "chosen_anchor"
        materialize_chosen_anchor(
            dataset, anchors, loaded, temperatures, sessions, selective_root, completed, store, seeds, folds,
        )
        selective_manifest["selected_count"] = len(completed)
        selective_manifest["expected_principal_count"] = len(completed)
    write_json(selective_root / "export_manifest.json", selective_manifest)
    points = point_table(store, seeds, folds)
    points.to_csv(selective_root / "point_summary.csv", index=False)
    decisions = decision_table(
        store, split, seeds, folds, args.boot, cmau_name, esau_name, anchor_model, not late_fusion_note,
    )
    decisions.to_csv(selective_root / "decision_table.csv", index=False)
    notes = {
        "anchor_rule": "per fold, higher seed-averaged development macro-F1; audio-only wins an exact tie",
        "tau_rule": "grid 0.30..0.70 step 0.05, maximum development macro-F1, exact ties take the larger tau",
        "logit_weight": "w multiplies audio-only z/T; ties take the larger w",
        "qmf": "energy=-logsumexp(z/T); weights=softmax(-energy); z=sum weight*(z/T); no training",
        "visual_evidence": "video_kept_tokens/32; 0 for zero_video and branch_disabled_video",
        "late_fusion": late_fusion_note or "computed",
        "decision_calibration": (
            "Macro-F1 uses argmax. VRR uses each system's own development temperature. "
            "Selective logits in the export are already temperature-scaled, so their raw "
            "probabilities are the protocol probabilities."
        ),
    }
    write_json(selective_root / "protocol_notes.json", notes)
    combined = build_combined(exports, manifest, selective_root, selective_manifest, out / "combined_export")
    pairs_path = out / "pairs.json"
    audio_name = audio_models[0] if audio_models else ""
    pairs_path.write_text(json.dumps(
        analysis_pairs(late_fusion_note, audio_name, anchor_model, cmau_name, esau_name), indent=2,
    ) + "\n", encoding="utf-8")
    analysis_dir = out / "analysis"
    from iemocap_mer.robust_stats import run_strong_analysis
    analysis = run_strong_analysis(
        combined, analysis_dir, None, "session", args.boot, args.hboot, False, pairs_path, "unstratified",
    )
    summary = {
        "selective_export": str(selective_root),
        "combined_export": str(combined),
        "analysis": analysis["out"],
        "anchors": {str(fold): anchors[fold] for fold in folds},
        "anchor_scores": {str(fold): anchor_scores[fold] for fold in folds},
        "late_fusion": late_fusion_note or "computed",
        "n_boot": args.boot,
        "n_hboot": args.hboot,
    }
    write_json(out / "summary.json", summary)
    print(f"Selective export: {selective_root}")
    print(f"Anchors: {summary['anchors']}")
    print(f"Late fusion: {summary['late_fusion']}")
    print(diagnostic_frame.groupby("selective_system")[["tau", "coverage_dev", "coverage_test"]].mean())
    print(points.to_string(index=False))
    print(decisions.to_string(index=False))
    print(f"Strong analysis: {analysis['out']}")
    return summary


def write_late_fusion(dataset, seed, fold, audio_name, video_name, weight, loaded, temperatures,
                      sessions, root, completed, store) -> None:
    del sessions
    audio_test = loaded[(audio_name, seed, fold)]["frames"]["test"]
    video_test = loaded[(video_name, seed, fold)]["frames"]["test"]
    conditions = sorted(set(audio_test.condition.astype(str)) | set(video_test.condition.astype(str)))
    audio_available = set(audio_test.condition.astype(str))
    video_available = set(video_test.condition.astype(str))
    dev_audio = condition_frame(loaded[(audio_name, seed, fold)]["frames"]["validation"], "clean")
    dev_video = align_to(dev_audio, condition_frame(loaded[(video_name, seed, fold)]["frames"]["validation"], "clean"))
    for system, fuse in (
        ("logit_avg_av", lambda a, ta, v, tv: averaged_logits(a, ta, v, tv, weight)),
        ("qmf_av", lambda a, ta, v, tv: qmf_fusion(a, ta, v, tv)[0]),
    ):
        run_id = f"{system}_fold_{fold}_seed{seed}"
        dev_z = fuse(
            logits_of(dev_audio), temperatures[(audio_name, seed, fold)],
            logits_of(dev_video), temperatures[(video_name, seed, fold)],
        )
        dev_rows = prediction_rows(
            dataset, system, seed, fold, run_id, "validation", "clean",
            dev_audio.utterance_id.astype(str), dev_audio.label.to_numpy(dtype=int), dev_z, dev_audio,
        )
        test_rows = []
        for condition in conditions:
            audio_used = condition if condition in audio_available else "clean"
            video_used = condition if condition in video_available else "clean"
            audio_rows = condition_frame(audio_test, audio_used)
            video_rows = align_to(audio_rows, condition_frame(video_test, video_used))
            fused = fuse(
                logits_of(audio_rows), temperatures[(audio_name, seed, fold)],
                logits_of(video_rows), temperatures[(video_name, seed, fold)],
            )
            ids = audio_rows.utterance_id.astype(str).to_numpy()
            labels = audio_rows.label.to_numpy(dtype=int)
            store[(system, int(seed), int(fold), condition)] = (ids, labels, np.asarray(fused, dtype=np.float64))
            source = video_rows if condition in video_available else audio_rows
            test_rows.extend(prediction_rows(
                dataset, system, seed, fold, run_id, "test", condition, ids, labels, fused, source,
            ))
        target = root / "runs" / run_id
        target.mkdir(parents=True)
        files = {
            "validation": write_table(target / "validation_predictions.csv.gz", dev_rows),
            "test": write_table(target / "test_predictions.csv.gz", test_rows),
        }
        record = {
            "run_id": run_id, "model": system, "seed": int(seed), "fold": int(fold),
            "history_status": "derived", "conditions": conditions, "files": files,
            "logit_avg_audio_weight": weight if system == "logit_avg_av" else None,
            "created": timestamp(),
        }
        write_json(target / "complete.json", record)
        completed.append(record)


def point_table(store, seeds, folds) -> pd.DataFrame:
    models = sorted({model for model, _seed, _fold, _condition in store})
    rows = []
    for model in models:
        for condition in ("clean", "zero_video"):
            pooled = pools_for(store, model, condition, seeds, folds)
            if pooled is None:
                continue
            scores, vrrs = [], []
            for ids, labels, clean_z, altered_z in pooled:
                del ids
                scores.append(macro_of(labels, clean_z if condition == "clean" else altered_z))
                if condition != "clean":
                    clean_p = softmax(clean_z)
                    altered_p = softmax(altered_z)
                    vrrs.append(float((altered_p.max(1) > clean_p.max(1)).mean()))
            rows.append({
                "model": model,
                "condition": condition,
                "macro_f1": float(np.mean(scores)),
                "vrr": float(np.mean(vrrs)) if vrrs else "",
            })
    return pd.DataFrame(rows)


def pools_for(store, model, condition, seeds, folds):
    """Per seed: ids, labels, clean logits, and logits of ``condition`` (clean duplicated)."""
    pools = []
    for seed in seeds:
        if any((model, seed, fold, "clean") not in store for fold in folds):
            return None
        if condition != "clean" and any((model, seed, fold, condition) not in store for fold in folds):
            return None
        chunks, altered = [], []
        for fold in folds:
            chunks.append(store[(model, seed, fold, "clean")])
            altered.append(store[(model, seed, fold, condition)])
        ids = np.concatenate([item[0] for item in chunks])
        order = np.argsort(ids, kind="mergesort")
        ids = ids[order]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate out-of-fold ids for {model} seed {seed}")
        labels = np.concatenate([item[1] for item in chunks])[order]
        clean_z = np.concatenate([item[2] for item in chunks])[order]
        other_ids = np.concatenate([item[0] for item in altered])
        other_order = np.argsort(other_ids, kind="mergesort")
        if not np.array_equal(other_ids[other_order], ids):
            raise ValueError(f"Condition ids differ for {model} {condition}")
        other_z = np.concatenate([item[2] for item in altered])[other_order]
        pools.append((ids, labels, clean_z, other_z))
    reference = pools[0][0]
    for ids, _labels, _clean, _other in pools[1:]:
        if not np.array_equal(ids, reference):
            raise ValueError(f"Seeds of {model} do not share utterance ids")
    return pools


def decision_table(store, split, seeds, folds, n_boot: int, cmau_name: str, esau_name: str,
                   anchor_model: str, late_fusion: bool) -> pd.DataFrame:
    test_meta = split[split["split"] == "test"].copy()
    if test_meta.utterance_id.duplicated().any():
        raise ValueError("Split manifest has duplicate test utterance ids")
    test_meta = test_meta.set_index("utterance_id", drop=False)
    comparisons = [
        (cmau_name, "selective_cmau", "selective_s1s2", True),
        (esau_name, "selective_esau", "selective_s1s2", True),
        (anchor_model, "selective_cmau", "selective_vs_anchor", False),
        (anchor_model, "selective_esau", "selective_vs_anchor", False),
    ]
    if late_fusion:
        comparisons.extend([
            ("logit_avg_av", "selective_cmau", "selective_vs_logit_avg", False),
            ("logit_avg_av", "selective_esau", "selective_vs_logit_avg", False),
            ("qmf_av", "selective_cmau", "selective_vs_qmf", False),
            ("qmf_av", "selective_esau", "selective_vs_qmf", False),
        ])
    pending = []
    for model_a, model_b, family, s_criteria in comparisons:
        clean_a = pools_for(store, model_a, "clean", seeds, folds)
        clean_b = pools_for(store, model_b, "clean", seeds, folds)
        if clean_a is None or clean_b is None:
            continue
        ids = clean_a[0][0]
        speaker_ids, strata, speaker_index = _speaker_axis(ids, test_meta, "session")
        mult = speaker_multiplicities(np.asarray(strata), n_boot, BOOT_SEED, mode="unstratified")
        clean_diff = contrast_macro(clean_a, clean_b, speaker_index, len(speaker_ids), mult, use_altered=False)
        fold_clean = fold_differences(store, model_a, model_b, "clean", seeds, folds, "macro_f1")
        pending.append(row_from_contrast(
            model_a, model_b, family, "clean", "macro_f1", clean_diff, fold_clean, s_criteria,
        ))
        zero_a = pools_for(store, model_a, "zero_video", seeds, folds)
        zero_b = pools_for(store, model_b, "zero_video", seeds, folds)
        if zero_a is None or zero_b is None:
            continue
        zero_diff = contrast_macro(zero_a, zero_b, speaker_index, len(speaker_ids), mult, use_altered=True)
        fold_zero = fold_differences(store, model_a, model_b, "zero_video", seeds, folds, "macro_f1")
        pending.append(row_from_contrast(
            model_a, model_b, family, "zero_video", "macro_f1", zero_diff, fold_zero, False,
        ))
        vrr_diff = contrast_vrr(zero_a, zero_b, speaker_index, len(speaker_ids), mult)
        fold_vrr = fold_differences(store, model_a, model_b, "zero_video", seeds, folds, "vrr")
        pending.append(row_from_contrast(
            model_a, model_b, family, "zero_video", "vrr", vrr_diff, fold_vrr, s_criteria,
        ))
    frame = pd.DataFrame(pending)
    if frame.empty:
        return frame
    frame["holm_t_p"] = np.nan
    for _key, index in frame.groupby(["family", "metric", "condition"], sort=False).groups.items():
        ordered = list(index)
        frame.loc[ordered, "holm_t_p"] = holm(frame.loc[ordered, "fold_t_p"].to_numpy(dtype=float))
    verdicts = []
    for item in frame.itertuples(index=False):
        verdicts.append(consensus_verdict(
            item.ci90_low, item.ci90_high, item.ci95_low, item.ci95_high,
            item.fold_t_ci90_low, item.fold_t_ci90_high, item.holm_t_p, item.metric,
        ))
    frame["consensus_verdict"] = verdicts
    frame["bootstrap_verdict"] = [
        bootstrap_component_verdict(item.ci90_low, item.ci90_high, item.ci95_low, item.ci95_high, item.metric)
        for item in frame.itertuples(index=False)
    ]
    frame["fold_t_verdict"] = [
        fold_component_verdict(item.fold_t_ci90_low, item.fold_t_ci90_high, item.holm_t_p, item.metric)
        for item in frame.itertuples(index=False)
    ]
    frame["s1_bootstrap"] = [
        bool(item.s_criteria and item.metric == "macro_f1" and item.condition == "clean"
             and item.effect_b_minus_a > 0 and _excludes(item.ci95_low, item.ci95_high))
        for item in frame.itertuples(index=False)
    ]
    by_pair = {(row.model_a, row.model_b): row for row in frame.itertuples(index=False)}
    s2_v1, s2_consensus = [], []
    for item in frame.itertuples(index=False):
        partner = by_pair.get((item.model_a, item.model_b))
        vrr_row = None
        macro_row = None
        for candidate in frame.itertuples(index=False):
            if (candidate.model_a, candidate.model_b) != (item.model_a, item.model_b):
                continue
            if candidate.metric == "vrr" and candidate.condition == "zero_video":
                vrr_row = candidate
            if candidate.metric == "macro_f1" and candidate.condition == "clean":
                macro_row = candidate
        applies = bool(item.s_criteria and item.metric == "macro_f1" and item.condition == "clean" and vrr_row)
        inside = applies and _inside(macro_row.ci90_low, macro_row.ci90_high)
        reduced = applies and vrr_row.effect_b_minus_a < 0 and _excludes(vrr_row.ci95_low, vrr_row.ci95_high)
        s2_v1.append(bool(inside and reduced))
        both_equivalent = applies and macro_row.consensus_verdict == "equivalent"
        vrr_different = applies and vrr_row.consensus_verdict == "different" and vrr_row.effect_b_minus_a < 0
        s2_consensus.append(bool(both_equivalent and vrr_different))
        del partner
    frame["s2_bootstrap"] = s2_v1
    frame["s2_consensus"] = s2_consensus
    return frame


def _excludes(low, high) -> bool:
    return bool(np.isfinite(low) and np.isfinite(high) and (low > 0.0 or high < 0.0))


def _inside(low, high, margin: float = 0.02) -> bool:
    return bool(np.isfinite(low) and np.isfinite(high) and low > -margin and high < margin)


def row_from_contrast(model_a, model_b, family, condition, metric, contrast, fold_diff, s_criteria) -> dict:
    tested = paired_t_test(fold_diff)
    intervals = percentile_intervals(contrast["samples"])
    return {
        "model_a": model_a,
        "model_b": model_b,
        "family": family,
        "condition": condition,
        "metric": metric,
        "effect_b_minus_a": contrast["effect"],
        **intervals,
        "bootstrap_p": contrast["p"],
        "fold_t_effect": tested["effect"],
        "fold_t_p": tested["p"],
        "fold_t_ci90_low": tested["ci90_low"],
        "fold_t_ci90_high": tested["ci90_high"],
        "fold_t_ci95_low": tested["ci95_low"],
        "fold_t_ci95_high": tested["ci95_high"],
        "s_criteria": bool(s_criteria and metric in {"macro_f1", "vrr"} and (
            (metric == "macro_f1" and condition == "clean") or (metric == "vrr" and condition == "zero_video")
        )),
    }


def contrast_macro(pools_a, pools_b, speaker_index, n_speakers, multiplicities, use_altered: bool) -> dict:
    from iemocap_mer.robust_stats import bootstrap_pvalue

    def stats_for(pools):
        rows = []
        for ids, labels, clean_z, altered_z in pools:
            del ids
            logits = altered_z if use_altered else clean_z
            parts = utterance_parts(labels, logits, 1.0)
            rows.append(accumulate_speakers(parts, speaker_index, n_speakers))
        return {key: np.stack([row[key] for row in rows], axis=0) for key in rows[0]}

    stats_a, stats_b = stats_for(pools_a), stats_for(pools_b)
    samples = (
        metrics_from_speaker_stats(stats_b, multiplicities)["macro_f1"]
        - metrics_from_speaker_stats(stats_a, multiplicities)["macro_f1"]
    )
    effect = float(
        metrics_from_speaker_stats(stats_b)["macro_f1"] - metrics_from_speaker_stats(stats_a)["macro_f1"]
    )
    return {"effect": effect, "samples": samples, "p": bootstrap_pvalue(samples)}


def contrast_vrr(pools_a, pools_b, speaker_index, n_speakers, multiplicities) -> dict:
    from iemocap_mer.robust_stats import bootstrap_pvalue

    def stats_for(pools):
        rows = []
        for ids, labels, clean_z, altered_z in pools:
            del ids
            clean = utterance_parts(labels, clean_z, 1.0)
            altered = utterance_parts(labels, altered_z, 1.0)
            one = accumulate_effects(clean, altered, speaker_index, n_speakers)
            rows.append({key: value for key, value in one.items() if not key.startswith("u_")})
        return {key: np.stack([row[key] for row in rows], axis=0) for key in rows[0]}

    stats_a, stats_b = stats_for(pools_a), stats_for(pools_b)
    samples = effect_rates(stats_b, multiplicities)["vrr"] - effect_rates(stats_a, multiplicities)["vrr"]
    effect = float(effect_rates(stats_b)["vrr"] - effect_rates(stats_a)["vrr"])
    return {"effect": effect, "samples": samples, "p": bootstrap_pvalue(samples)}


def fold_differences(store, model_a, model_b, condition, seeds, folds, metric: str) -> np.ndarray:
    differences = []
    for fold in folds:
        per_seed = []
        for seed in seeds:
            _ids, labels, clean_z = store[(model_a, seed, fold, "clean")]
            _ids_b, labels_b, clean_b = store[(model_b, seed, fold, "clean")]
            if condition == "clean":
                left_z, right_z = clean_z, clean_b
                left_y, right_y = labels, labels_b
            else:
                _alt_ids, left_y, left_z = store[(model_a, seed, fold, condition)]
                _alt_ids_b, right_y, right_z = store[(model_b, seed, fold, condition)]
            if metric == "macro_f1":
                per_seed.append(macro_of(right_y, right_z) - macro_of(left_y, left_z))
            else:
                per_seed.append(vrr_of(labels_b, clean_b, right_z) - vrr_of(labels, clean_z, left_z))
        differences.append(float(np.mean(per_seed)))
    return np.asarray(differences, dtype=np.float64)


def vrr_of(labels, clean_z, altered_z) -> float:
    del labels
    return float((softmax(altered_z).max(1) > softmax(clean_z).max(1)).mean())


def analysis_pairs(late_fusion_note: str, audio_name: str, anchor_model: str,
                   cmau_name: str, esau_name: str) -> list:
    pairs = [[left, right, "iemocap_historical_posthoc"] for left, right in PRIMARY_PAIRS]
    pairs.extend([
        [cmau_name, "selective_cmau", "selective_s1s2"],
        [esau_name, "selective_esau", "selective_s1s2"],
        [anchor_model, "selective_cmau", "selective_vs_anchor"],
        [anchor_model, "selective_esau", "selective_vs_anchor"],
    ])
    if not late_fusion_note and audio_name:
        pairs.extend([
            [audio_name, "logit_avg_av", "late_fusion_posthoc"],
            [audio_name, "qmf_av", "late_fusion_posthoc"],
            [cmau_name, "logit_avg_av", "late_fusion_posthoc"],
            [cmau_name, "qmf_av", "late_fusion_posthoc"],
            ["logit_avg_av", "selective_cmau", "selective_vs_logit_avg"],
            ["logit_avg_av", "selective_esau", "selective_vs_logit_avg"],
            ["qmf_av", "selective_cmau", "selective_vs_qmf"],
            ["qmf_av", "selective_esau", "selective_vs_qmf"],
        ])
    return pairs


def materialize_chosen_anchor(dataset, anchors, loaded, temperatures, sessions, root, completed, store,
                              seeds, folds) -> None:
    """Fold-wise anchor as one system when the chosen anchor is not the same model on every fold.

    Files keep the anchor's raw logits so a later temperature fit is unchanged. The in-memory
    store keeps the temperature-scaled logits already used for the protocol probabilities.
    """
    del temperatures, sessions
    for fold in folds:
        anchor_name = anchors[fold]
        for seed in seeds:
            for (stored_model, stored_seed, stored_fold, condition), payload in list(store.items()):
                if stored_model == anchor_name and stored_seed == int(seed) and stored_fold == int(fold):
                    store[("chosen_anchor", int(seed), int(fold), condition)] = payload
            anchor_test = loaded[(anchor_name, seed, fold)]["frames"]["test"]
            dev = condition_frame(loaded[(anchor_name, seed, fold)]["frames"]["validation"], "clean")
            run_id = f"chosen_anchor_fold_{fold}_seed{seed}"
            dev_rows = prediction_rows(
                dataset, "chosen_anchor", seed, fold, run_id, "validation", "clean",
                dev.utterance_id.astype(str), dev.label.to_numpy(dtype=int), logits_of(dev), dev,
            )
            conditions = sorted(set(anchor_test.condition.astype(str)))
            test_rows = []
            for condition in conditions:
                group = condition_frame(anchor_test, condition)
                ids = group.utterance_id.astype(str).to_numpy()
                labels = group.label.to_numpy(dtype=int)
                test_rows.extend(prediction_rows(
                    dataset, "chosen_anchor", seed, fold, run_id, "test", condition,
                    ids, labels, logits_of(group), group,
                ))
            target = root / "runs" / run_id
            target.mkdir(parents=True)
            files = {
                "validation": write_table(target / "validation_predictions.csv.gz", dev_rows),
                "test": write_table(target / "test_predictions.csv.gz", test_rows),
            }
            record = {
                "run_id": run_id, "model": "chosen_anchor", "seed": int(seed), "fold": int(fold),
                "history_status": "derived", "conditions": conditions, "files": files,
                "anchor": anchor_name, "created": timestamp(),
            }
            write_json(target / "complete.json", record)
            completed.append(record)


def build_combined(exports: Path, original: dict, selective_root: Path, selective: dict, destination: Path) -> Path:
    destination.mkdir(parents=True)
    (destination / "runs").mkdir()
    shutil.copyfile(exports / "split_manifest.csv", destination / "split_manifest.csv")
    runs = []
    sources = [(exports, original["runs"]), (selective_root, selective["runs"])]
    for root, records in sources:
        for record in records:
            shutil.copytree(root / "runs" / record["run_id"], destination / "runs" / record["run_id"])
            runs.append(record)
    settings = {
        "dataset": original.get("dataset", selective.get("dataset")),
        "split_manifest_sha256": sha256(destination / "split_manifest.csv"),
        "combined_from": [str(exports), str(selective_root)],
    }
    payload = {
        "format_version": 1,
        "dataset": original.get("dataset", "iemocap"),
        "settings": settings,
        "complete": True,
        "selected_count": len(runs),
        "expected_principal_count": len(runs),
        "history_unknown_count": sum(run.get("history_status") == "history_unknown" for run in runs),
        "runs": runs,
        "failures": [],
        "updated": timestamp(),
    }
    write_json(destination / "export_manifest.json", payload)
    return destination


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        selective_pilot(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
