#!/usr/bin/env python3
"""Standalone companion; point --repo to the existing private project clone."""
from __future__ import annotations
import argparse
import sys

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit", help="Check caches, partitions and all 75 checkpoint identities")
    a.add_argument("--repo", required=True); a.add_argument("--out", required=True)
    e = sub.add_parser("export", help="Re-evaluate selected checkpoints without training")
    e.add_argument("--repo", required=True); e.add_argument("--audit", required=True); e.add_argument("--out", required=True)
    e.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    e.add_argument("--batch-size", type=int, default=16)
    e.add_argument("--corruption-seed", type=int, default=20260926)
    e.add_argument("--models", nargs="+")
    e.add_argument("--max-runs", type=int)
    e.add_argument("--include-history-unknown", action="store_true",
                   help="Export for flagged re-evaluation despite missing historical counters; never allows rejected runs")
    e.add_argument("--resume", action="store_true")
    n = sub.add_parser("analyze", help="Score paired exports, calibrate on development and aggregate")
    n.add_argument("--exports", required=True); n.add_argument("--out", required=True)
    n.add_argument("--allow-partial", action="store_true", help="Diagnostic analysis of a smoke/subset export")
    s = sub.add_parser("smoke-test", help="Test actual project forward passes and disabled-branch invariance")
    s.add_argument("--repo", required=True); s.add_argument("--out", required=True)
    s.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    r = sub.add_parser("ravdess-manifest", help="Prepare proposed RAVDESS AV inventory/splits only")
    r.add_argument("--raw", required=True); r.add_argument("--out", required=True)
    z = sub.add_parser("pack", help="Package the evidence folder; exclude feature/model formats")
    z.add_argument("--outputs", required=True); z.add_argument("--out", required=True)
    return p

def main():
    args = parser().parse_args()
    try:
        if args.command == "audit":
            from merkit.audit import run_audit
            return run_audit(args)
        if args.command == "export":
            if args.batch_size < 1:
                raise ValueError("--batch-size must be positive")
            from merkit.export import run_export
            return run_export(args)
        if args.command == "analyze":
            from merkit.analysis import run_analysis
            return run_analysis(args)
        from merkit import support
        return {"smoke-test": support.run_smoke, "ravdess-manifest": support.run_ravdess_manifest,
                "pack": support.run_pack}[args.command](args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
