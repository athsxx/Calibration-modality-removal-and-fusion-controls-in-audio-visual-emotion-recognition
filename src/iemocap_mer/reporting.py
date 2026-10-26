from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data import EMOTION4_LABELS

SCALAR_METRICS = (
    "accuracy", "uar", "macro_f1", "weighted_f1",
    "valence_ccc", "activation_ccc", "va_mae",
)

BASE_MODELS = ("audio", "video", "concat", "cross_attention", "essence")
ROBUSTNESS_CONDITIONS = ("clean", "missing_audio", "missing_video", "noisy_audio", "masked_video")


def _load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_run_name(run_name: str) -> dict:
    for fold in (f"fold_{i}" for i in range(1, 6)):
        suffix = f"_{fold}"
        if run_name.endswith(f"{suffix}_no_audio_to_video"):
            return {"model": "essence", "fold": fold, "variant": "no_audio_to_video"}
        if run_name.endswith(f"{suffix}_no_video_to_audio"):
            return {"model": "essence", "fold": fold, "variant": "no_video_to_audio"}
        for model in BASE_MODELS:
            if run_name == f"{model}{suffix}":
                return {"model": model, "fold": fold, "variant": "full"}
    return {"model": run_name, "fold": "", "variant": "full"}


def aggregate_runs(checkpoint_root: str | Path, output_csv: str | Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(checkpoint_root).glob("*.metrics.json")):
        if path.name.startswith("._") or path.name.startswith("smoke_"):
            continue
        payload = _load_metrics(path)
        run_name = path.name.removesuffix(".metrics.json")
        row = {"run": run_name, "best_epoch": payload["best_epoch"]}
        row.update({f"test_{key}": payload["test"].get(key) for key in SCALAR_METRICS})
        rows.append(row)
    runs = pd.DataFrame(rows)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_csv, index=False)
    if not runs.empty:
        runs["model"] = runs.run.str.extract(
            r"^(audio|video|concat|cross_attention|essence)"
        )[0]
        scalar_columns = [column for column in runs if column.startswith("test_")]
        summary = runs.groupby("model")[scalar_columns].agg(["mean", "std"])
        summary.to_csv(output_csv.with_name(output_csv.stem + "_summary.csv"))
    return runs


def _model_summary_table(checkpoint_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(checkpoint_root.glob("*.metrics.json")):
        if path.name.startswith("._") or path.name.startswith("smoke_"):
            continue
        run_name = path.name.removesuffix(".metrics.json")
        meta = _parse_run_name(run_name)
        if meta["variant"] != "full" or meta["model"] not in BASE_MODELS:
            continue
        payload = _load_metrics(path)
        row = {"run": run_name, **meta, "best_epoch": payload["best_epoch"]}
        for key in SCALAR_METRICS:
            row[f"test_{key}"] = payload["test"].get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def _per_class_recall_table(checkpoint_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(checkpoint_root.glob("*.metrics.json")):
        if path.name.startswith("._") or path.name.startswith("smoke_"):
            continue
        run_name = path.name.removesuffix(".metrics.json")
        meta = _parse_run_name(run_name)
        if meta["variant"] != "full" or meta["model"] not in BASE_MODELS:
            continue
        recall = _load_metrics(path)["test"].get("per_class_recall", {})
        for label in EMOTION4_LABELS:
            rows.append(
                {
                    "run": run_name,
                    "model": meta["model"],
                    "fold": meta["fold"],
                    "emotion": label,
                    "recall": recall.get(label),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return (
        frame.groupby(["model", "emotion"])["recall"]
        .agg(["mean", "std"])
        .reset_index()
    )


def _essence_ablation_table(checkpoint_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(checkpoint_root.glob("essence_fold_*.metrics.json")):
        if path.name.startswith("._"):
            continue
        run_name = path.name.removesuffix(".metrics.json")
        meta = _parse_run_name(run_name)
        payload = _load_metrics(path)
        rows.append(
            {
                "fold": meta["fold"],
                "variant": meta["variant"],
                "best_epoch": payload["best_epoch"],
                "test_macro_f1": payload["test"]["macro_f1"],
                "test_uar": payload["test"]["uar"],
            }
        )
    return pd.DataFrame(rows)


def _aggregate_robustness(results_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_root.glob("essence_fold_*_robustness.json")):
        if path.name.startswith("._"):
            continue
        fold = path.name.removeprefix("essence_").removesuffix("_robustness.json")
        payload = _load_metrics(path)
        clean_f1 = payload["clean"]["macro_f1"]
        for condition in ROBUSTNESS_CONDITIONS:
            metrics = payload[condition]
            rows.append(
                {
                    "fold": fold,
                    "condition": condition,
                    "macro_f1": metrics["macro_f1"],
                    "uar": metrics["uar"],
                    "macro_f1_drop": metrics.get("macro_f1_drop", clean_f1 - metrics["macro_f1"]),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    summary = (
        frame.groupby("condition")[["macro_f1", "uar", "macro_f1_drop"]]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def _aggregate_explanations(results_root: Path) -> dict:
    frames = []
    for path in sorted(results_root.glob("essence_fold_*_explanations.csv")):
        if path.name.startswith("._"):
            continue
        fold = path.name.removeprefix("essence_").removesuffix("_explanations.csv")
        frame = pd.read_csv(path)
        frame["fold"] = fold
        frames.append(frame)
    if not frames:
        return {}
    merged = pd.concat(frames, ignore_index=True)
    return {
        "samples": int(len(merged)),
        "mean_audio_reliability": float(merged["audio_reliability"].mean()),
        "mean_video_reliability": float(merged["video_reliability"].mean()),
        "mean_confidence_drop_without_audio": float(merged["confidence_drop_without_audio"].mean()),
        "mean_confidence_drop_without_video": float(merged["confidence_drop_without_video"].mean()),
        "audio_dominant_fraction": float(
            (merged["confidence_drop_without_audio"] > merged["confidence_drop_without_video"]).mean()
        ),
    }


def _aggregate_distress_proxy(results_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_root.glob("essence_fold_*_distress_proxy.json")):
        if path.name.startswith("._"):
            continue
        fold = path.name.removeprefix("essence_").removesuffix("_distress_proxy.json")
        payload = _load_metrics(path)
        for entry in payload.get("threshold_sensitivity", []):
            rows.append({"fold": fold, **entry})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    group_cols = ["valence_threshold_normalized", "activation_threshold_normalized"]
    return (
        frame.groupby(group_cols)[["precision", "recall", "f1", "positive_predictions"]]
        .agg(["mean", "std"])
        .reset_index()
    )


def _research_conclusions(summary: pd.DataFrame, robustness: pd.DataFrame, ablations: pd.DataFrame) -> list[str]:
    bullets: list[str] = []
    if summary.empty:
        return bullets

    ranking = (
        summary.groupby("model")["test_macro_f1"]
        .mean()
        .sort_values(ascending=False)
    )
    bullets.append(
        f"Best average test macro-F1 among base models: {ranking.index[0]} ({ranking.iloc[0]:.3f})."
    )
    if "audio" in ranking.index and "concat" in ranking.index:
        bullets.append(
            f"Audio-only ({ranking.get('audio', float('nan')):.3f}) vs concat ({ranking.get('concat', float('nan')):.3f}) "
            "shows whether multimodal fusion adds value over speech alone."
        )
    if "essence" in ranking.index and "concat" in ranking.index:
        delta = ranking.get("essence", 0) - ranking.get("concat", 0)
        if delta < 0:
            bullets.append(
                f"Essence transfer underperformed concat by {abs(delta):.3f} macro-F1 on average; "
                "directional transfer needs revision or stronger video features."
            )
        else:
            bullets.append("Essence transfer beat concat on average macro-F1.")

    if not robustness.empty:
        for _, row in robustness.iterrows():
            condition = row["condition"]
            if condition == "clean":
                continue
            drop = row["macro_f1_drop_mean"]
            if pd.notna(drop) and drop > 0.05:
                bullets.append(f"Robustness: {condition} reduces macro-F1 by ~{drop:.3f} on average.")
            elif pd.notna(drop) and drop < -0.02:
                bullets.append(
                    f"Robustness: {condition} slightly improves macro-F1 (~{abs(drop):.3f}), "
                    "suggesting the removed modality may add noise."
                )

    if not ablations.empty:
        full = ablations[ablations.variant == "full"].groupby("fold")["test_macro_f1"].mean()
        no_a2v = ablations[ablations.variant == "no_audio_to_video"].set_index("fold")["test_macro_f1"]
        no_v2a = ablations[ablations.variant == "no_video_to_audio"].set_index("fold")["test_macro_f1"]
        if not no_a2v.empty:
            delta = (full - no_a2v.reindex(full.index)).mean()
            bullets.append(
                f"Disabling audio→video distillation changes macro-F1 by {delta:+.3f} on average across folds."
            )
        if not no_v2a.empty:
            delta = (full - no_v2a.reindex(full.index)).mean()
            bullets.append(
                f"Disabling video→audio distillation changes macro-F1 by {delta:+.3f} on average across folds."
            )

    bullets.append(
        "Distress-proxy outputs are exploratory VA-derived analyses only; they are not clinical distress detection."
    )
    return bullets


def _write_markdown_report(report: dict, output_path: Path) -> None:
    lines = [
        "# IEMOCAP MER Research Report",
        "",
        "## Model comparison (test macro-F1 / UAR, mean ± std)",
        "",
    ]
    for row in report["model_ranking"]:
        lines.append(
            f"- **{row['model']}**: macro-F1 {row['macro_f1_mean']:.3f} ± {row['macro_f1_std']:.3f}, "
            f"UAR {row['uar_mean']:.3f} ± {row['uar_std']:.3f} ({row['folds']} folds)"
        )
    lines.extend(["", "## Research conclusions", ""])
    lines.extend(f"- {bullet}" for bullet in report["conclusions"])
    lines.extend(["", "## Essence ablations (test macro-F1 by fold)", ""])
    if report["essence_ablations"]:
        lines.append("| fold | full | no_a2v | no_v2a |")
        lines.append("|---|---:|---:|---:|")
        for fold, values in report["essence_ablations"].items():
            lines.append(
                f"| {fold} | {values.get('full', '—')} | {values.get('no_audio_to_video', '—')} | "
                f"{values.get('no_video_to_audio', '—')} |"
            )
    lines.extend(["", "## Robustness (essence, macro-F1 drop vs clean)", ""])
    for row in report["robustness_summary"]:
        lines.append(
            f"- **{row['condition']}**: macro-F1 {row['macro_f1_mean']:.3f}, "
            f"drop {row['macro_f1_drop_mean']:+.3f}"
        )
    if report.get("explanation_summary"):
        exp = report["explanation_summary"]
        lines.extend(
            [
                "",
                "## Modality contribution (essence explanations)",
                "",
                f"- Mean audio reliability: {exp['mean_audio_reliability']:.3f}",
                f"- Mean video reliability: {exp['mean_video_reliability']:.3f}",
                f"- Mean confidence drop without audio: {exp['mean_confidence_drop_without_audio']:.3f}",
                f"- Mean confidence drop without video: {exp['mean_confidence_drop_without_video']:.3f}",
                f"- Fraction where audio removal hurts more than video: {exp['audio_dominant_fraction']:.3f}",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_video_quality(
    artifacts_root: str | Path,
    manifest_path: str | Path,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest.emotion4.notna() & manifest.emotion4.ne("")].copy()
    rows = []
    for video_dir in ("video", "video_v2"):
        quality_path = Path(artifacts_root) / "features" / f"{video_dir}_quality.csv"
        if not quality_path.exists():
            continue
        quality = pd.read_csv(quality_path).merge(
            manifest[["utterance_id", "session", "emotion4"]], on="utterance_id", how="inner"
        )
        rate_col = "face_track_rate" if "face_track_rate" in quality.columns else "face_detection_rate"
        for session, group in quality.groupby("session"):
            rows.append(
                {
                    "video_dir": video_dir,
                    "session": int(session),
                    "utterances": int(len(group)),
                    "mean_face_track_rate": float(group[rate_col].mean()),
                    "zero_track_rate": int((group[rate_col] == 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def generate_research_report(
    checkpoint_root: str | Path,
    results_root: str | Path,
    output_dir: str | Path,
) -> dict:
    checkpoint_root = Path(checkpoint_root)
    results_root = Path(results_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_runs(checkpoint_root, output_dir / "cross_fold_metrics.csv")
    summary = _model_summary_table(checkpoint_root)
    per_class = _per_class_recall_table(checkpoint_root)
    ablations = _essence_ablation_table(checkpoint_root)
    robustness = _aggregate_robustness(results_root)
    explanations = _aggregate_explanations(results_root)
    distress = _aggregate_distress_proxy(results_root)

    ranking = []
    if not summary.empty:
        for model, group in summary.groupby("model"):
            ranking.append(
                {
                    "model": model,
                    "folds": int(len(group)),
                    "macro_f1_mean": float(group["test_macro_f1"].mean()),
                    "macro_f1_std": float(group["test_macro_f1"].std(ddof=0)),
                    "uar_mean": float(group["test_uar"].mean()),
                    "uar_std": float(group["test_uar"].std(ddof=0)),
                }
            )
        ranking.sort(key=lambda row: row["macro_f1_mean"], reverse=True)

    ablation_pivot: dict[str, dict[str, float | None]] = {}
    if not ablations.empty:
        for fold, group in ablations.groupby("fold"):
            ablation_pivot[fold] = {
                row.variant: round(float(row.test_macro_f1), 3) for row in group.itertuples()
            }

    robustness_rows = []
    if not robustness.empty:
        for _, row in robustness.iterrows():
            robustness_rows.append(
                {
                    "condition": row["condition"],
                    "macro_f1_mean": float(row["macro_f1_mean"]),
                    "macro_f1_drop_mean": float(row["macro_f1_drop_mean"]),
                }
            )

    report = {
        "model_ranking": ranking,
        "conclusions": _research_conclusions(summary, robustness, ablations),
        "essence_ablations": ablation_pivot,
        "robustness_summary": robustness_rows,
        "explanation_summary": explanations,
    }

    per_class.to_csv(output_dir / "per_class_recall_summary.csv", index=False)
    ablations.to_csv(output_dir / "essence_ablation_summary.csv", index=False)
    robustness.to_csv(output_dir / "robustness_summary.csv", index=False)
    distress.to_csv(output_dir / "distress_proxy_summary.csv", index=False)
    (output_dir / "research_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown_report(report, output_dir / "research_report.md")
    return report
