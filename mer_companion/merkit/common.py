from __future__ import annotations
import csv
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_COMMIT = "db0ad3491103f1a63feb559373cdb178f4683d10"
LABELS = ("angry", "happy_excited", "neutral", "sad")
SEEDS = (0, 1, 7)
FOLDS = (1, 2, 3, 4, 5)
SPECS = {
    "audio_w2v": ("audio", "w2v_seed{seed}", "configs/audio_w2v.yaml", "video"),
    "concatmatched_v2": ("concat_matched", "w2v_v2_seed{seed}", "configs/w2v_video_v2.yaml", "video_v2"),
    "essence_v2": ("essence", "w2v_v2_stage2fix_seed{seed}", "configs/w2v_video_v2.yaml", "video_v2"),
    "concatmatched_au": ("concat_matched", "w2v_au_seed{seed}", "configs/w2v_video_au.yaml", "video_au"),
    "essence_au": ("essence", "w2v_au_seed{seed}", "configs/w2v_video_au.yaml", "video_au"),
}
CONDITIONS = ("clean", "zero_audio", "zero_video", "branch_disabled_audio",
              "branch_disabled_video", "video_token_loss_25", "video_token_loss_50",
              "video_token_loss_75")
PRIMARY_PAIRS = (
    ("audio_w2v", "concatmatched_v2"), ("concatmatched_v2", "essence_v2"),
    ("audio_w2v", "concatmatched_au"), ("audio_w2v", "essence_au"),
    ("concatmatched_au", "essence_au"), ("concatmatched_v2", "concatmatched_au"),
    ("essence_v2", "essence_au"),
)

def timestamp():
    return datetime.now(timezone.utc).isoformat()

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v
                             for k, v in row.items()})

def new_output(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path

def within(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes the expected root: {relative}")
    return path

def git_info(repo):
    def run(*args):
        p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else "unavailable"
    return {"commit": run("rev-parse", "HEAD"), "status": run("status", "--short"),
            "python": sys.version, "platform": platform.platform()}

def toolkit_hashes():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.py"))
            if "__pycache__" not in p.parts}

def repo_import(repo):
    root = Path(repo).resolve()
    if not (root / "src/iemocap_mer").is_dir():
        raise ValueError("--repo must point to distress_ready_mer (containing src/iemocap_mer)")
    sys.path.insert(0, str(root / "src"))
    return root

def expected_runs():
    rows = []
    for model, (name, tag, config, video) in SPECS.items():
        for seed in SEEDS:
            for fold in FOLDS:
                stem = f"{name}_fold_{fold}_{tag.format(seed=seed)}"
                rows.append(dict(model=model, model_name=name, seed=seed, fold=fold,
                    run_id=stem, checkpoint_relative_path=f"artifacts/checkpoints/{stem}.pt",
                    config_relative_path=config, audio_dir="audio_w2v", video_dir=video))
    return rows

def iemocap_identity(uid):
    m = re.fullmatch(r"(Ses\d{2})[FM]_.+_([FM])\d+", str(uid))
    if not m:
        raise ValueError(f"Unexpected IEMOCAP utterance ID: {uid}")
    return int(m[1][3:]), m[1] + m[2]

def feature_relative(cache, uid):
    session, _ = iemocap_identity(uid)
    return f"artifacts/features/{cache}/Session{session}/{uid}.pt"

def validate_splits(frame, folds):
    """The unit of leakage checking is a speaker, not just an utterance."""
    if frame.utterance_id.duplicated().any():
        raise ValueError("Duplicate utterance IDs in the manifest")
    ids = set(frame.utterance_id)
    lookup = frame.set_index("utterance_id")
    records, seen_test = [], set()
    for fold in FOLDS:
        allocation = folds[f"fold_{fold}"]
        groups = {part: set(allocation[part]) for part in ("train", "dev", "test")}
        if any(not group for group in groups.values()):
            raise ValueError(f"Empty partition in fold {fold}")
        if any(len(allocation[p]) != len(groups[p]) for p in groups):
            raise ValueError(f"Duplicate split IDs in fold {fold}")
        if set.union(*groups.values()) != ids:
            raise ValueError(f"Split coverage differs from manifest in fold {fold}")
        parts = list(groups)
        for i, a in enumerate(parts):
            for b in parts[i + 1:]:
                if groups[a] & groups[b]:
                    raise ValueError(f"Utterance leakage in fold {fold}")
                sa = {iemocap_identity(u)[1] for u in groups[a]}
                sb = {iemocap_identity(u)[1] for u in groups[b]}
                if sa & sb:
                    raise ValueError(f"Speaker leakage in fold {fold}")
        if seen_test & groups["test"]:
            raise ValueError("A test utterance occurs in more than one outer fold")
        seen_test |= groups["test"]
        for part in parts:
            for uid in allocation[part]:
                session, speaker = iemocap_identity(uid)
                records.append(dict(dataset="iemocap", fold=fold,
                    split="validation" if part == "dev" else part, utterance_id=uid,
                    session=session, speaker_id=speaker,
                    label=LABELS.index(lookup.loc[uid, "emotion4"])))
    if seen_test != ids:
        raise ValueError("Outer test folds do not cover the manifest exactly")
    return records

def metadata_eligibility(spec, checkpoint, metrics):
    rejected, unknown = [], []
    if checkpoint.get("model_name") != spec["model_name"]:
        rejected.append("checkpoint architecture does not match selection")
    if checkpoint.get("audio_dir") != "audio_w2v":
        rejected.append("checkpoint audio cache is not audio_w2v")
    if spec["model_name"] != "audio" and checkpoint.get("video_dir") != spec["video_dir"]:
        rejected.append("checkpoint visual cache does not match selection")
    if not isinstance(checkpoint.get("epoch"), int) or checkpoint["epoch"] < 1:
        rejected.append("checkpoint epoch is missing or invalid")
    if checkpoint.get("epoch") != metrics.get("best_epoch"):
        rejected.append("checkpoint epoch and metrics best_epoch differ")
    if metrics.get("seed") is None:
        unknown.append("seed not recorded in metrics")
    elif metrics["seed"] != spec["seed"]:
        rejected.append("metrics seed differs from selection")
    if metrics.get("aborted_nonfinite") is True:
        rejected.append("run aborted for non-finite values")
    count = metrics.get("nonfinite_batch_count")
    if count is None:
        unknown.append("historical non-finite batch count is not recorded")
    elif not isinstance(count, (int, float)) or count != 0:
        rejected.append("non-finite batch count is not zero")
    if spec["model_name"] == "essence":
        if checkpoint.get("stage") != 2:
            rejected.append("saved Essence checkpoint is not Stage 2")
        for key, expected in (("stage2_reached", True), ("used_stage1_fallback", False)):
            if key not in metrics:
                unknown.append(f"{key} not recorded")
            elif metrics[key] is not expected:
                rejected.append(f"{key} is not {expected}")
    if not isinstance(metrics.get("test"), dict) or "macro_f1" not in metrics.get("test", {}):
        rejected.append("test macro-F1 is missing")
    else:
        value = metrics["test"]["macro_f1"]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            rejected.append("stored test macro-F1 is invalid")
    status = "rejected" if rejected else "history_unknown" if unknown else "eligible"
    return status, rejected, unknown

def applicable_conditions(model_name):
    return ("clean", "zero_audio") if model_name == "audio" else CONDITIONS
