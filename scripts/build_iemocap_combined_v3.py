#!/usr/bin/env python3
"""Combined IEMOCAP export view for protocol-v3 selective fusion.

Symlinks the runs the pilot needs (audio, video-only AU, ConcatMatched-AU,
Essence-AU) and writes a new manifest. Original export directories are not
modified.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NEEDED = {
    "exports": ("audio_w2v", "concatmatched_au", "essence_au"),
    "exports_iemocap_video_au": ("video_au",),
}


def main() -> int:
    destination = REPO / "mtap_outputs" / "exports_iemocap_combined_v3"
    if destination.exists():
        print(f"Refusing to overwrite {destination}", file=sys.stderr)
        return 2
    runs = []
    split_source = None
    for folder, models in NEEDED.items():
        root = REPO / "mtap_outputs" / folder
        manifest = json.loads((root / "export_manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("complete", False):
            raise ValueError(f"{folder} is not a complete export")
        if split_source is None:
            split_source = root / "split_manifest.csv"
        else:
            other = (root / "split_manifest.csv").read_bytes()
            if other != split_source.read_bytes():
                raise ValueError("IEMOCAP split manifests differ")
        wanted = set(models)
        found = [run for run in manifest["runs"] if run["model"] in wanted]
        if {run["model"] for run in found} != wanted:
            raise ValueError(f"{folder} is missing {[name for name in wanted if name not in {r['model'] for r in found}]}")
        runs.extend((root, run) for run in found)
    destination.mkdir(parents=True)
    (destination / "runs").mkdir()
    shutil.copyfile(split_source, destination / "split_manifest.csv")
    copied = []
    for root, run in runs:
        target = destination / "runs" / run["run_id"]
        if target.exists():
            raise ValueError(f"Duplicate run id {run['run_id']}")
        target.symlink_to(root / "runs" / run["run_id"], target_is_directory=True)
        copied.append(run)
    payload = {
        "format_version": 1,
        "dataset": "iemocap",
        "settings": {
            "combined_from": [str(REPO / "mtap_outputs" / folder) for folder in NEEDED],
            "models": sorted({run["model"] for run in copied}),
            "note": "Symlinks only. Original exports were not modified.",
        },
        "complete": True,
        "selected_count": len(copied),
        "expected_principal_count": len(copied),
        "history_unknown_count": sum(run.get("history_status") == "history_unknown" for run in copied),
        "runs": copied,
        "failures": [],
    }
    (destination / "export_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {destination} with {len(copied)} runs: {payload['settings']['models']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
