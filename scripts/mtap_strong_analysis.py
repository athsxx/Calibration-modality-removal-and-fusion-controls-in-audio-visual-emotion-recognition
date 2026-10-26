#!/usr/bin/env python3
"""Speaker-bootstrap analysis of an MER logit export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iemocap_mer.robust_stats import run_strong_analysis


def parser() -> argparse.ArgumentParser:
    tool = argparse.ArgumentParser(description=__doc__)
    tool.add_argument("--exports", required=True)
    tool.add_argument("--out", required=True)
    tool.add_argument("--temperatures")
    tool.add_argument("--strata-column", default="session")
    tool.add_argument("--boot", type=int, default=10000)
    tool.add_argument("--hboot", type=int, default=2000)
    tool.add_argument("--resampling", choices=("unstratified", "stratified"), default="unstratified")
    tool.add_argument("--allow-partial", action="store_true")
    tool.add_argument("--pairs-json")
    return tool


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        summary = run_strong_analysis(
            args.exports, args.out, args.temperatures, args.strata_column,
            args.boot, args.hboot, args.allow_partial, args.pairs_json, args.resampling,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Strong analysis saved: {summary['out']} ({summary['n_runs']} runs)")
    print(
        f"partial={summary['partial']} history_unknown={summary['history_unknown_count']} "
        f"B={summary['B_speaker']} hboot={summary['B_hierarchical']} "
        f"resampling={summary['resampling']} sensitivity={summary['sensitivity_resampling']}"
    )
    for model, row in summary["clean_raw_macro_f1"].items():
        print(
            f"{model} clean/raw macro_f1={row['estimate']:.16f} "
            f"CI95=[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"
        )
    if summary["skipped_pairs"]:
        print("Skipped pairs:")
        for row in summary["skipped_pairs"]:
            print(f"  {row['model_a']} vs {row['model_b']}: {row['reason']}")
    if summary["gee_failures"]:
        print("GEE failures: " + "; ".join(summary["gee_failures"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
