#!/usr/bin/env python3
"""Combined export view for Step 9 controls.

Copies selected models from existing exports into a new folder. The strong
analysis rejects paths that resolve outside the export root, so the view
stores copies. Source exports are not modified and an existing destination
is refused.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

COLUMNS = ["dataset", "fold", "split", "utterance_id", "session", "speaker_id", "label"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def split_rows(path: Path) -> list[tuple[str, ...]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(column not in reader.fieldnames for column in COLUMNS):
            raise ValueError(f"{path} is missing split columns")
        return [tuple(row[column] for column in COLUMNS) for row in reader]


def load_manifest(root: Path) -> dict:
    manifest = json.loads((root / "export_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete", False):
        raise ValueError(f"{root} is not a complete export")
    return manifest


def main(argv: list[str] | None = None) -> int:
    tool = argparse.ArgumentParser(description=__doc__)
    tool.add_argument("--include", nargs=2, action="append", metavar=("EXPORT", "MODEL"), required=True)
    tool.add_argument("--out", required=True)
    args = tool.parse_args(argv)
    if len(args.include) < 2:
        print("Need at least two export/model pairs", file=sys.stderr)
        return 2

    destination = Path(args.out).resolve()
    if destination.exists():
        print(f"Refusing to overwrite {destination}", file=sys.stderr)
        return 2

    sources = []
    reference_rows = None
    for export, model in args.include:
        root = Path(export).resolve()
        rows = split_rows(root / "split_manifest.csv")
        if reference_rows is None:
            reference_rows = rows
        elif sorted(rows) != sorted(reference_rows):
            raise ValueError(f"Split manifest content differs: {root}")
        manifest = load_manifest(root)
        found = [run for run in manifest["runs"] if run["model"] == model]
        if len(found) != 15:
            raise ValueError(f"{root} has {len(found)} runs for {model}, expected 15")
        sources.append((root, found))

    destination.mkdir(parents=True)
    (destination / "runs").mkdir()
    baseline_split = Path(args.include[0][0]).resolve() / "split_manifest.csv"
    shutil.copyfile(baseline_split, destination / "split_manifest.csv")
    copied = []
    for root, runs in sources:
        for run in runs:
            target = destination / "runs" / run["run_id"]
            if target.exists():
                raise ValueError(f"Duplicate run id {run['run_id']}")
            shutil.copytree(root / "runs" / run["run_id"], target)
            copied.append(run)
    payload = {
        "format_version": 1,
        "dataset": load_manifest(Path(args.include[0][0]).resolve()).get("dataset"),
        "settings": {
            "combined_from": [str(Path(export).resolve()) for export, _model in args.include],
            "models": [model for _export, model in args.include],
            "split_manifest_sha256": sha256(destination / "split_manifest.csv"),
            "note": "Copies of the compared models. Source exports were not modified.",
        },
        "complete": True,
        "selected_count": len(copied),
        "expected_principal_count": len(copied),
        "history_unknown_count": sum(run.get("history_status") == "history_unknown" for run in copied),
        "runs": copied,
        "failures": [],
    }
    (destination / "export_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {destination} with {len(copied)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
