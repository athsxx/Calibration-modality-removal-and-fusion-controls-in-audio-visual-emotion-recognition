#!/usr/bin/env python3
"""Draw MTAP handoff figures from existing analysis CSVs.

Reads mtap_outputs/analysis_strong_v2, analysis_strong_ravdess, and
analysis_strong_cremad (and the CML analysis folders). Writes only
mtap_outputs/figures. Does not train, export, or modify those analysis
folders.

Figures are drawn at the sn-jnl single-column text width (31 pc) so that
an 8 pt font is 8 pt when the PDF is included at \\textwidth. No titles
are drawn inside the images.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "mtap_outputs"
FIG = OUT / "figures"

# sn-jnl.cls single-column geometry: text={31pc, 194.25mm}.
TEXT_WIDTH_IN = 31 * 12 / 72.27

# Okabe–Ito, colour-blind-safe. Consensus colours are the same on every forest.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
VERDICT_COLOR = {
    "different": BLUE,
    "equivalent": GREEN,
    "inconclusive": ORANGE,
}

IEMOCAP_MODELS = [
    "audio_w2v",
    "concatmatched_v2",
    "essence_v2",
    "concatmatched_au",
    "essence_au",
]
IEMOCAP_LABELS = [
    "Audio",
    "ConcatMatched-ResNet",
    "Essence-ResNet",
    "ConcatMatched-AU",
    "Essence-AU",
]
CORPUS_MODELS = [
    "audio",
    "video_resnet",
    "video_au",
    "concat_matched_resnet",
    "essence_resnet",
    "concat_matched_au",
    "essence_au",
]
CORPUS_LABELS = [
    "Audio",
    "Video-ResNet",
    "Video-AU",
    "ConcatMatched-ResNet",
    "Essence-ResNet",
    "ConcatMatched-AU",
    "Essence-AU",
]
SOURCE_FIELDS = [
    "figure",
    "series",
    "model",
    "condition",
    "calibration",
    "metric",
    "estimate",
    "ci95_low",
    "ci95_high",
]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
        }
    )


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def flt(value: str):
    if value is None or value == "":
        return None
    return float(value)


def one(rows: list[dict], **kwargs) -> dict:
    found = [row for row in rows if all(row.get(key) == value for key, value in kwargs.items())]
    if len(found) != 1:
        raise SystemExit(f"expected 1 row for {kwargs}, found {len(found)}")
    return found[0]


def record(bucket: list[dict], figure: str, series: str, row: dict) -> None:
    bucket.append(
        {
            "figure": figure,
            "series": series,
            "model": row["model"],
            "condition": row["condition"],
            "calibration": row["calibration"],
            "metric": row["metric"],
            "estimate": row["estimate"],
            "ci95_low": row.get("ci95_low", ""),
            "ci95_high": row.get("ci95_high", ""),
        }
    )


def errors(estimate: float, row: dict) -> tuple[float, float]:
    low = flt(row.get("ci95_low", ""))
    high = flt(row.get("ci95_high", ""))
    return (
        0.0 if low is None else estimate - low,
        0.0 if high is None else high - estimate,
    )


def save(fig, stem: str) -> None:
    """Write a PDF whose media box matches the print width, plus a 300 dpi PNG."""
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png", dpi=300)
    plt.close(fig)


def lookup(rows: list[dict], **kwargs) -> dict:
    found = [row for row in rows if all(row.get(key) == value for key, value in kwargs.items())]
    if len(found) != 1:
        raise SystemExit(f"expected 1 row for {kwargs}, found {len(found)}")
    return found[0]


def verdict_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", color=color, label=name, linestyle="")
        for name, color in VERDICT_COLOR.items()
    ]


CML_CORPORA = (
    ("IEMOCAP", "analysis_strong_v2", "analysis_cml_iemocap", "concatmatched_au"),
    ("RAVDESS", "analysis_strong_ravdess", "analysis_cml_ravdess", "concat_matched_au"),
    ("CREMA-D", "analysis_strong_cremad", "analysis_cml_cremad", "concat_matched_au"),
)

CALIBRATION_SERIES = (
    ("baseline", "raw", BLUE, "ConcatMatched-AU, raw"),
    ("cml", "raw", ORANGE, "CML-inspired, raw"),
    ("baseline", "temperature", GREEN, "ConcatMatched-AU, temperature"),
    ("cml", "temperature", VERMILLION, "CML-inspired, temperature"),
)


def calibration_2x2_figure(source: list[dict]) -> None:
    """Four conditions by corpus for ECE and both zero-fill VRR panels.

    ConcatMatched-AU and the CML-inspired penalty, each raw and
    temperature-scaled. Both systems are read from analysis_cml_*.
    """
    panels = (
        ("clean", "ece_width", "point", "Clean equal-width ECE", (0.0, 0.26)),
        ("zero_video", "vrr", "effect", "Zero-video VRR", (0.0, 0.55)),
        ("zero_audio", "vrr", "effect", "Zero-audio VRR", (0.0, 0.55)),
    )
    fig, axes = plt.subplots(3, 1, figsize=(TEXT_WIDTH_IN, 5.45), sharex=True)
    xs = list(range(len(CML_CORPORA)))
    width = 0.18
    legend_handles = None
    for ax, (condition, metric, kind, ylabel, ylim) in zip(axes, panels):
        for slot, (which, calibration, color, label) in enumerate(CALIBRATION_SERIES):
            offset = (slot - 1.5) * width
            values, low_err, high_err = [], [], []
            for corpus, _strong, cml_folder, always_name in CML_CORPORA:
                model = "concat_matched_au_cml" if which == "cml" else always_name
                table = "point_and_ci_metrics.csv" if kind == "point" else "condition_effects.csv"
                row = lookup(
                    read_csv(OUT / cml_folder / table),
                    model=model,
                    condition=condition,
                    calibration=calibration,
                    metric=metric,
                )
                estimate = float(row["estimate"])
                low, high = errors(estimate, row)
                values.append(estimate)
                low_err.append(low)
                high_err.append(high)
                record(source, "calibration_2x2", f"{corpus}:{label}:{metric}:{condition}", row)
            bars = ax.bar(
                [x + offset for x in xs],
                values,
                width,
                label=label,
                color=color,
                zorder=2,
                yerr=[low_err, high_err],
                capsize=2,
                ecolor="#000000",
                error_kw={"elinewidth": 0.6, "capthick": 0.6, "zorder": 3},
            )
            if legend_handles is None:
                legend_handles = []
            if len(legend_handles) < len(CALIBRATION_SERIES):
                legend_handles.append(bars)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_axisbelow(True)
    axes[-1].set_xticks(xs, [item[0] for item in CML_CORPORA])
    fig.legend(
        legend_handles,
        [item[3] for item in CALIBRATION_SERIES],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.56, 0.995),
    )
    fig.subplots_adjust(left=0.18, right=0.98, top=0.84, bottom=0.07, hspace=0.28)
    save(fig, "calibration_2x2")


FOREST_SPECS = (
    ("macro-F1", "analysis", "CMAU − audio", "clean", "raw", "macro_f1", "audio", "concat_au"),
    ("macro-F1", "analysis", "EAU − audio", "clean", "raw", "macro_f1", "audio", "essence_au"),
    ("macro-F1", "analysis", "vAU − audio", "clean", "raw", "macro_f1", "audio", "video_au"),
    ("macro-F1", "analysis", "EAU − CMAU", "clean", "raw", "macro_f1", "concat_au", "essence_au"),
    ("macro-F1", "selective", "selective − CMAU", "clean", "raw", "macro_f1", "concat_au", "selective_cmau"),
    ("macro-F1", "cml", "CML − CMAU", "clean", "raw", "macro_f1", "concat_au", "cml"),
    ("reliability", "cml", "zero-video VRR", "zero_video", "temperature", "vrr", "concat_au", "cml"),
    ("reliability", "cml", "zero-audio VRR", "zero_audio", "temperature", "vrr", "concat_au", "cml"),
    ("reliability", "cml", "clean ECE", "clean", "temperature", "ece_width", "concat_au", "cml"),
)

NAMES = {
    "IEMOCAP": {"audio": "audio_w2v", "concat_au": "concatmatched_au", "essence_au": "essence_au", "video_au": None, "selective_cmau": "selective_cmau", "cml": "concat_matched_au_cml"},
    "RAVDESS": {"audio": "audio", "concat_au": "concat_matched_au", "essence_au": "essence_au", "video_au": "video_au", "selective_cmau": "selective_cmau", "cml": "concat_matched_au_cml"},
    "CREMA-D": {"audio": "audio", "concat_au": "concat_matched_au", "essence_au": "essence_au", "video_au": "video_au", "selective_cmau": "selective_cmau", "cml": "concat_matched_au_cml"},
}
FOLDERS = {
    ("IEMOCAP", "analysis"): "analysis_strong_v2",
    ("RAVDESS", "analysis"): "analysis_strong_ravdess",
    ("CREMA-D", "analysis"): "analysis_strong_cremad",
    ("IEMOCAP", "cml"): "analysis_cml_iemocap",
    ("RAVDESS", "cml"): "analysis_cml_ravdess",
    ("CREMA-D", "cml"): "analysis_cml_cremad",
    ("IEMOCAP", "selective"): "selective_v3_iemocap/analysis",
    ("RAVDESS", "selective"): "selective_v3_ravdess/analysis",
    ("CREMA-D", "selective"): "selective_v3_cremad/analysis",
}


def forest_points(source: list[dict]) -> dict[str, list[dict]]:
    cached: dict[str, list[dict]] = {}
    grouped: dict[str, list[dict]] = {"macro-F1": [], "reliability": []}
    for panel, kind, label, condition, calibration, metric, left, right in FOREST_SPECS:
        for corpus in ("IEMOCAP", "RAVDESS", "CREMA-D"):
            model_a = NAMES[corpus][left]
            model_b = NAMES[corpus][right]
            if model_a is None or model_b is None:
                continue
            folder = FOLDERS[(corpus, kind)]
            if folder not in cached:
                cached[folder] = read_csv(OUT / folder / "paired_comparisons.csv")
            row = lookup(
                cached[folder],
                model_a=model_a,
                model_b=model_b,
                condition=condition,
                calibration=calibration,
                metric=metric,
            )
            grouped[panel].append(
                {
                    "label": f"{corpus}  {label}",
                    "effect": float(row["effect_b_minus_a"]),
                    "low": float(row["ci95_low"]),
                    "high": float(row["ci95_high"]),
                    "verdict": row["consensus_verdict"],
                }
            )
            record(
                source,
                "cross_corpus_forest",
                f"{corpus}:{label}",
                {
                    "model": f"{model_a}->{model_b}",
                    "condition": condition,
                    "calibration": calibration,
                    "metric": metric,
                    "estimate": row["effect_b_minus_a"],
                    "ci95_low": row["ci95_low"],
                    "ci95_high": row["ci95_high"],
                },
            )
    return grouped


def draw_forest(points: list[dict], stem: str, xlabel: str) -> None:
    n = len(points)
    height = 0.24 * n + 0.95
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, height))
    ys = list(range(n))[::-1]
    for y, item in zip(ys, points):
        color = VERDICT_COLOR[item["verdict"]]
        ax.errorbar(
            item["effect"],
            y,
            xerr=[[item["effect"] - item["low"]], [item["high"] - item["effect"]]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2,
            markersize=4,
            elinewidth=0.8,
            capthick=0.8,
        )
    ax.axvline(0, color="#000000", lw=0.6)
    ax.set_yticks(ys, [item["label"] for item in points])
    ax.set_ylim(-0.7, n - 0.3)
    lows = [item["low"] for item in points]
    highs = [item["high"] for item in points]
    span = max(highs) - min(lows)
    pad = 0.05 * span if span else 0.02
    ax.set_xlim(min(0.0, min(lows)) - pad, max(0.0, max(highs)) + pad)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", length=0)
    fig.legend(
        handles=verdict_handles(),
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.62, 0.995),
    )
    fig.subplots_adjust(left=0.40, right=0.98, top=0.90, bottom=0.14)
    save(fig, stem)


def forest_figure(source: list[dict]) -> None:
    grouped = forest_points(source)
    draw_forest(
        grouped["macro-F1"],
        "cross_corpus_forest_f1",
        "Clean macro-F1 effect and 95% interval",
    )
    draw_forest(
        grouped["reliability"],
        "cross_corpus_forest_reliability",
        "Effect (CML − ConcatMatched-AU) and 95% interval",
    )


FAITHFUL_FOLDERS = {
    "IEMOCAP": "analysis_cmlf_iemocap",
    "RAVDESS": "analysis_cmlf_ravdess",
    "CREMA-D": "analysis_cmlf_cremad",
}
FAITHFUL_PANELS = (
    ("cmlfaithful_vs_concat_au", "faithful_cml_vs_concat", "Effect (faithful − ConcatMatched-AU) and 95% interval"),
    ("cmlfaithful_vs_cml_penalty", "faithful_cml_vs_penalty", "Effect (faithful − CML-inspired) and 95% interval"),
)
FAITHFUL_ROWS = (
    ("clean", "raw", "macro_f1", "clean macro-F1"),
    ("zero_audio", "temperature", "vrr", "zero-audio VRR"),
    ("zero_video", "temperature", "vrr", "zero-video VRR"),
    ("branch_disabled_audio", "temperature", "vrr", "branch-audio VRR"),
    ("branch_disabled_video", "temperature", "vrr", "branch-video VRR"),
    ("clean", "raw", "ece_width", "ECE, raw"),
    ("clean", "temperature", "ece_width", "ECE, temperature"),
)


def faithful_cml_figure(source: list[dict]) -> None:
    """One print-width forest per comparator. Does not redraw earlier figures."""
    cached: dict[str, list[dict]] = {}
    for family, stem, xlabel in FAITHFUL_PANELS:
        points = []
        for condition, calibration, metric, label in FAITHFUL_ROWS:
            for corpus, folder in FAITHFUL_FOLDERS.items():
                if folder not in cached:
                    cached[folder] = read_csv(OUT / folder / "paired_comparisons.csv")
                row = lookup(
                    cached[folder],
                    family=family,
                    condition=condition,
                    calibration=calibration,
                    metric=metric,
                )
                points.append(
                    {
                        "label": f"{corpus}  {label}",
                        "effect": float(row["effect_b_minus_a"]),
                        "low": float(row["ci95_low"]),
                        "high": float(row["ci95_high"]),
                        "verdict": row["consensus_verdict"],
                    }
                )
                record(
                    source,
                    "faithful_cml_effects",
                    f"{corpus}:{family}:{label}",
                    {
                        "model": f"{row['model_a']}->{row['model_b']}",
                        "condition": condition,
                        "calibration": calibration,
                        "metric": metric,
                        "estimate": row["effect_b_minus_a"],
                        "ci95_low": row["ci95_low"],
                        "ci95_high": row["ci95_high"],
                    },
                )
        draw_forest(points, stem, xlabel)


def label_pad(fig, labels) -> float:
    """Left margin, as a fraction of the print width, that fits the y labels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widest = 0.0
    for lab in labels:
        artist = fig.text(0, 0, lab, fontsize=8)
        widest = max(widest, artist.get_window_extent(renderer=renderer).width)
        artist.remove()
    width_in = widest / fig.dpi
    return min(0.58, (width_in + 0.28) / TEXT_WIDTH_IN)


def horizontal_bars(labels, series, xlim, xlabel, stem) -> None:
    """series: list of (values, low_err, high_err, color, legend_label)."""
    n = len(labels)
    height = 0.46 * n + (1.25 if len(series) > 1 else 0.85)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, height))
    ys = list(range(n))
    bar_h = 0.62 if len(series) == 1 else 0.36
    for index, (values, low_err, high_err, color, legend) in enumerate(series):
        if len(series) == 1:
            offset = 0.0
        else:
            offset = (index - 0.5) * bar_h
        ax.barh(
            [y + offset for y in ys],
            values,
            height=bar_h * 0.92,
            color=color,
            label=legend,
            zorder=2,
            xerr=[low_err, high_err],
            capsize=2,
            ecolor="#000000",
            error_kw={"elinewidth": 0.6, "capthick": 0.6, "zorder": 3},
        )
    ax.set_yticks(ys, labels)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", length=0)
    left = label_pad(fig, labels)
    if len(series) > 1:
        ax.legend(
            loc="lower center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 1.04),
        )
        fig.subplots_adjust(left=left, right=0.98, top=0.84, bottom=0.12)
    else:
        fig.subplots_adjust(left=left, right=0.98, top=0.97, bottom=0.16)
    save(fig, stem)


def collect_macro(points, models):
    estimates, low_err, high_err, rows = [], [], [], []
    for model in models:
        row = one(points, model=model, condition="clean", calibration="raw", metric="macro_f1")
        estimate = float(row["estimate"])
        low, high = errors(estimate, row)
        estimates.append(estimate)
        low_err.append(low)
        high_err.append(high)
        rows.append(row)
    return estimates, low_err, high_err, rows


def main() -> None:
    apply_style()
    source: list[dict] = []
    points = read_csv(OUT / "analysis_strong_v2" / "point_and_ci_metrics.csv")
    effects = read_csv(OUT / "analysis_strong_v2" / "condition_effects.csv")

    estimates, low_err, high_err, rows = collect_macro(points, IEMOCAP_MODELS)
    for row in rows:
        record(source, "iemocap_clean_macro_f1", "iemocap", row)
    horizontal_bars(
        IEMOCAP_LABELS,
        [(estimates, low_err, high_err, BLUE, "IEMOCAP")],
        (0.48, 0.68),
        "Clean macro-F1",
        "iemocap_clean_macro_f1",
    )

    raw_ece, scaled_ece = [], []
    raw_ece_lo, raw_ece_hi, scaled_ece_lo, scaled_ece_hi = [], [], [], []
    for model in IEMOCAP_MODELS:
        raw = one(points, model=model, condition="clean", calibration="raw", metric="ece_width")
        scaled = one(points, model=model, condition="clean", calibration="temperature", metric="ece_width")
        raw_est = float(raw["estimate"])
        scaled_est = float(scaled["estimate"])
        raw_ece.append(raw_est)
        scaled_ece.append(scaled_est)
        lo, hi = errors(raw_est, raw)
        raw_ece_lo.append(lo)
        raw_ece_hi.append(hi)
        lo, hi = errors(scaled_est, scaled)
        scaled_ece_lo.append(lo)
        scaled_ece_hi.append(hi)
        record(source, "iemocap_ece_width", "raw", raw)
        record(source, "iemocap_ece_width", "temperature", scaled)
    # These two IEMOCAP charts are not placed in the manuscript. They are
    # regenerated so every file this script writes shares the print style.
    horizontal_bars(
        IEMOCAP_LABELS,
        [
            (raw_ece, raw_ece_lo, raw_ece_hi, BLUE, "Raw"),
            (scaled_ece, scaled_ece_lo, scaled_ece_hi, GREEN, "Temperature"),
        ],
        (0.0, 0.30),
        "15-bin equal-width ECE",
        "iemocap_ece_width",
    )

    raw_vrr, scaled_vrr, vrr_labels = [], [], []
    raw_vrr_lo, raw_vrr_hi, scaled_vrr_lo, scaled_vrr_hi = [], [], [], []
    for model, label in zip(IEMOCAP_MODELS, IEMOCAP_LABELS):
        condition = "zero_audio" if model == "audio_w2v" else "zero_video"
        raw = one(effects, model=model, condition=condition, calibration="raw", metric="vrr")
        scaled = one(effects, model=model, condition=condition, calibration="temperature", metric="vrr")
        raw_est = float(raw["estimate"])
        scaled_est = float(scaled["estimate"])
        raw_vrr.append(raw_est)
        scaled_vrr.append(scaled_est)
        lo, hi = errors(raw_est, raw)
        raw_vrr_lo.append(lo)
        raw_vrr_hi.append(hi)
        lo, hi = errors(scaled_est, scaled)
        scaled_vrr_lo.append(lo)
        scaled_vrr_hi.append(hi)
        short = "zero-audio" if condition == "zero_audio" else "zero-video"
        vrr_labels.append(f"{label} ({short})")
        record(source, "iemocap_vrr_calibration", "raw", raw)
        record(source, "iemocap_vrr_calibration", "temperature", scaled)
    horizontal_bars(
        vrr_labels,
        [
            (raw_vrr, raw_vrr_lo, raw_vrr_hi, BLUE, "Raw"),
            (scaled_vrr, scaled_vrr_lo, scaled_vrr_hi, ORANGE, "Temperature"),
        ],
        (0.0, 1.0),
        "Violating-ranking rate",
        "iemocap_vrr_calibration",
    )

    series = []
    for folder, label, color in (
        ("analysis_strong_ravdess", "RAVDESS", BLUE),
        ("analysis_strong_cremad", "CREMA-D", ORANGE),
    ):
        corpus_points = read_csv(OUT / folder / "point_and_ci_metrics.csv")
        values, low_e, high_e, rows = collect_macro(corpus_points, CORPUS_MODELS)
        for row in rows:
            record(source, "ravdess_cremad_clean_macro_f1", label, row)
        series.append((values, low_e, high_e, color, label))
    horizontal_bars(
        CORPUS_LABELS,
        series,
        (0.48, 0.90),
        "Clean macro-F1",
        "ravdess_cremad_clean_macro_f1",
    )

    calibration_2x2_figure(source)
    forest_figure(source)
    faithful_cml_figure(source)

    with (FIG / "figure_source_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_FIELDS)
        writer.writeheader()
        writer.writerows(source)
    print(f"Wrote {FIG}")


if __name__ == "__main__":
    main()
