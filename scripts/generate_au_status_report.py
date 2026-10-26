#!/usr/bin/env python3
"""Generate paper/visual_backbone_au_status.md from completed AU pipeline artifacts."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
PAPER = ROOT / "paper"


def _parse_times(log_path: Path) -> dict[str, str]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, str] = {}
    for key, pattern in (
        ("extract_start", r"RESUME_START\s+(\S+)"),
        ("extract_end", r"RESUME_END\s+(\S+)"),
        ("train_start", r"TRAIN_START\s+(\S+)"),
        ("pipeline_end", r"PIPELINE_END\s+(\S+)"),
    ):
        match = re.search(pattern, text)
        if match:
            out[key] = match.group(1)
    return out


def _wall_clock_hours(times: dict[str, str]) -> float | None:
    if "extract_start" not in times:
        return None
    end_key = "pipeline_end" if "pipeline_end" in times else "extract_end"
    if end_key not in times:
        return None
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    start = datetime.strptime(times["extract_start"], fmt).replace(tzinfo=timezone.utc)
    end = datetime.strptime(times[end_key], fmt).replace(tzinfo=timezone.utc)
    return (end - start).total_seconds() / 3600.0


def _fold_metrics(tag: str | None) -> pd.DataFrame:
    rows = []
    for path in sorted((ART / "checkpoints").glob("*.metrics.json")):
        name = path.stem.replace(".metrics", "")
        if tag:
            if not name.endswith(f"_{tag}"):
                continue
        elif "_au" in name or "_w2v_au" in name:
            pass
        for model, suffix in (
            ("video", "_au"),
            ("concat_matched", "_w2v_au"),
            ("essence", "_w2v_au"),
        ):
            pattern = f"{model}_fold_"
            if pattern in name and name.endswith(suffix):
                payload = json.loads(path.read_text(encoding="utf-8"))
                fold = re.search(r"fold_\d", name).group(0)
                rows.append(
                    {
                        "model": model,
                        "fold": fold,
                        "macro_f1": payload["test"].get("macro_f1"),
                        "uar": payload["test"].get("uar"),
                    }
                )
    return pd.DataFrame(rows)


def _session_face_rates(quality_csv: Path, manifest_csv: Path) -> pd.DataFrame:
    quality = pd.read_csv(quality_csv)
    manifest = pd.read_csv(manifest_csv)[["utterance_id", "session"]]
    merged = quality.merge(manifest, on="utterance_id", how="left")
    rate_col = "face_track_rate" if "face_track_rate" in merged.columns else "face_detection_rate"
    summary = (
        merged.groupby("session")[rate_col]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "face_track_rate", "count": "utterances"})
    )
    summary["face_track_rate_pct"] = (summary["face_track_rate"] * 100).round(1)
    return summary


def main() -> None:
    progress = ART / "reports" / "video_au_progress.log"
    extract_log = ART / "reports" / "video_au_extraction.log"
    quality = ART / "features" / "video_au_quality.csv"
    manifest = ART / "manifests" / "iemocap_utterances.csv"
    verify = ART / "features" / "video_verification_report.json"
    table1 = PAPER / "tables" / "table1_model_comparison.md"

    times = _parse_times(extract_log)
    hours = _wall_clock_hours(times)
    metrics = _fold_metrics(None)

    n_pt = len(list((ART / "features" / "video_au").rglob("*.pt")))
    verify_info = {}
    if verify.exists():
        verify_info = json.loads(verify.read_text(encoding="utf-8"))

    lines = [
        "# Visual backbone AU status (py-feat Action Units)",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Tooling",
        "",
        "- **Package:** `py-feat` (PyPI `py-feat`, imports `from feat import Detectorv1`)",
        "- **OpenFace C++:** not installed (`which OpenFace FeatureExtraction` → not found)",
        "- **Rationale:** pip-installable AU detection (RetinaFace + MobileFaceNet landmarks + XGBoost AU classifiers) without compiling OpenFace; 20 OpenFace-style AU intensity columns per frame.",
        "- **Face crop:** reuses v2 pipeline (CLAHE + Haar frontal/profile cascade, speaker half-frame, 0.35 margin); crops resized to 224×224 before py-feat.",
        "- **Note:** `pip install py-feat` upgraded torch 2.3.1→2.13.0 and torchaudio 2.3.1→2.11.0 in the active conda env.",
        "",
        "## Feature dimensionality",
        "",
        "- **Per frame:** 20 AU intensities (`AU01`…`AU43`, see `features_au.py` `AU_COLUMNS`)",
        "- **Per utterance tensor:** `[num_frames, 20]` float32 (max 32 frames @ 6 fps)",
        f"- **Cache:** `artifacts/features/video_au/` — {n_pt} `.pt` files on disk",
        "",
    ]

    if verify_info:
        lines += [
            f"- **Verification:** {verify_info.get('valid', '?')}/{verify_info.get('expected', 5531)} valid",
            "",
        ]

    if quality.exists() and manifest.exists():
        session_rates = _session_face_rates(quality, manifest)
        lines += ["## Face-track rate by session (AU pipeline)", "", "| Session | utterances | face-track rate |", "|---:|---:|---:|"]
        for _, row in session_rates.sort_values("session").iterrows():
            lines.append(
                f"| {int(row.session)} | {int(row.utterances)} | {row.face_track_rate_pct:.1f}% |"
            )
        overall = pd.read_csv(quality)
        rate_col = "face_track_rate" if "face_track_rate" in overall.columns else "face_detection_rate"
        lines += [
            "",
            f"**Overall mean face-track rate:** {overall[rate_col].mean()*100:.1f}% ({len(overall)} utterances)",
            "",
        ]

    if hours is not None:
        lines += [f"## Wall-clock time", "", f"- **Pipeline (extract → train/analyze):** {hours:.2f} hours", ""]
    if progress.exists():
        lines += ["### Extraction progress log (tail)", "", "```", *progress.read_text(encoding="utf-8").strip().splitlines()[-8:], "```", ""]

    if not metrics.empty:
        lines += ["## Results (5-fold LOSO macro-F1 / UAR)", ""]
        for model, label in (
            ("video", "Video-only (AU)"),
            ("concat_matched", "ConcatMatched (W2V+AU)"),
            ("essence", "Essence (W2V+AU)"),
        ):
            sub = metrics[metrics.model == model]
            if sub.empty:
                continue
            f1_mean, f1_std = sub.macro_f1.mean(), sub.macro_f1.std()
            uar_mean, uar_std = sub.uar.mean(), sub.uar.std()
            lines.append(
                f"- **{label}:** macro-F1 {f1_mean:.3f} ± {f1_std:.3f}, UAR {uar_mean:.3f} ± {uar_std:.3f} ({len(sub)} folds)"
            )
        lines += [
            "",
            "### Comparison vs ResNet-18 video (from `paper/tables/table1_model_comparison.md`)",
            "",
            "| Model | Macro-F1 (mean ± std) | UAR (mean ± std) |",
            "|---|---:|---:|",
        ]
        if table1.exists():
            for row in table1.read_text(encoding="utf-8").splitlines():
                if row.startswith("|") and ("Video (v" in row or "ConcatMatched (W2V+v2)" in row or "Essence (Wav2Vec2 + v2)" in row):
                    lines.append(row)
        for model, label in (
            ("video", "Video-only (AU, py-feat)"),
            ("concat_matched", "ConcatMatched (W2V+AU)"),
            ("essence", "Essence (W2V+AU)"),
        ):
            sub = metrics[metrics.model == model]
            if sub.empty:
                continue
            f1_mean, f1_std = sub.macro_f1.mean(), sub.macro_f1.std()
            uar_mean, uar_std = sub.uar.mean(), sub.uar.std()
            lines.append(
                f"| {label} | {len(sub)} | {f1_mean:.3f} ± {f1_std:.3f} | {uar_mean:.3f} ± {uar_std:.3f} |"
            )
        lines.append("")

    out = PAPER / "visual_backbone_au_status.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
