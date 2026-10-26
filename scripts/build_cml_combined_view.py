#!/usr/bin/env python3
"""Combined export view for CML versus always-on ConcatMatched-AU.

Copies the two models' run directories into a new folder. The strong analysis
resolves paths and rejects files that land outside the export root, so this
view stores copies rather than symlinks. Source exports are not modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_manifest(root: Path) -> dict:
    manifest = json.loads((root / "export_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete", False):
        raise ValueError(f"{root} is not a complete export")
    return manifest


def main(argv: list[str] | None = None) -> int:
    tool = argparse.ArgumentParser(description=__doc__)
    tool.add_argument("--baseline-export", required=True)
    tool.add_argument("--baseline-model", required=True)
    tool.add_argument("--cml-export", required=True)
    tool.add_argument("--cml-model", default="concat_matched_au_cml")
    tool.add_argument("--out", required=True)
    tool.add_argument("--pairs-out", required=True)
    args = tool.parse_args(argv)

    baseline_root = Path(args.baseline_export).resolve()
    cml_root = Path(args.cml_export).resolve()
    destination = Path(args.out).resolve()
    if destination.exists():
        print(f"Refusing to overwrite {destination}", file=sys.stderr)
        return 2
    baseline_split = (baseline_root / "split_manifest.csv").read_bytes()
    cml_split = (cml_root / "split_manifest.csv").read_bytes()
    if baseline_split != cml_split:
        raise ValueError("Split manifests differ; the two exports cannot be pooled")

    selected = []
    for root, model in ((baseline_root, args.baseline_model), (cml_root, args.cml_model)):
        manifest = load_manifest(root)
        found = [run for run in manifest["runs"] if run["model"] == model]
        if len(found) != 15:
            raise ValueError(f"{root} has {len(found)} runs for {model}, expected 15")
        selected.extend((root, run) for run in found)

    destination.mkdir(parents=True)
    (destination / "runs").mkdir()
    (destination / "split_manifest.csv").write_bytes(baseline_split)
    copied = []
    for root, run in selected:
        target = destination / "runs" / run["run_id"]
        if target.exists():
            raise ValueError(f"Duplicate run id {run['run_id']}")
        shutil.copytree(root / "runs" / run["run_id"], target)
        copied.append(run)
    split_hash = sha256(destination / "split_manifest.csv")
    payload = {
        "format_version": 1,
        "dataset": load_manifest(baseline_root).get("dataset"),
        "settings": {
            "combined_from": [str(baseline_root), str(cml_root)],
            "models": [args.baseline_model, args.cml_model],
            "split_manifest_sha256": split_hash,
            "note": "Copies of the two compared models. Source exports were not modified.",
        },
        "complete": True,
        "selected_count": len(copied),
        "expected_principal_count": len(copied),
        "history_unknown_count": sum(run.get("history_status") == "history_unknown" for run in copied),
        "runs": copied,
        "failures": [],
    }
    (destination / "export_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    pairs = [{
        "model_a": args.baseline_model,
        "model_b": args.cml_model,
        "family": "cml_vs_concat_au",
    }]
    pairs_path = Path(args.pairs_out)
    if pairs_path.exists():
        print(f"Refusing to overwrite {pairs_path}", file=sys.stderr)
        return 2
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_path.write_text(json.dumps(pairs, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {destination} with {len(copied)} runs and {pairs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
