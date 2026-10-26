"""Replace CREMA-D Git LFS pointer files with their verified media for the four study classes.

The clone at ../CREMA-D has pointer files only and a corrupt pack, so `git lfs pull`
fails. GitHub serves each LFS object at media.githubusercontent.com; a file is accepted only if
its SHA-256 equals the oid recorded in its pointer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MEDIA_URL = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/{}"
CLASSES = ("ANG", "HAP", "NEU", "SAD")


def pointer(path: Path) -> tuple[str, int] | None:
    if path.stat().st_size > 1024:
        return None
    text = path.read_text(errors="replace")
    oid = re.search(r"oid sha256:([0-9a-f]{64})", text)
    size = re.search(r"size (\d+)", text)
    if not oid or not size:
        raise ValueError(f"Unrecognised small file: {path}")
    return oid.group(1), int(size.group(1))


def fetch(root: Path, relative: str, oid: str, size: int, retries: int = 6) -> str:
    target = root / relative
    temp = target.with_name(target.name + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(MEDIA_URL.format(relative), timeout=120) as response:
                data = response.read()
            if len(data) != size or hashlib.sha256(data).hexdigest() != oid:
                raise ValueError(f"hash/size mismatch ({len(data)} bytes)")
            temp.write_bytes(data)
            os.replace(temp, target)
            return "fetched"
        except Exception as exc:
            if attempt == retries - 1:
                return f"failed: {exc}"
            time.sleep(5 * (attempt + 1))
    return "failed"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="../CREMA-D")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--report", default="../CREMA-D/fetch_report.json")
    args = parser.parse_args()
    root = Path(args.root)
    jobs, already = [], 0
    for folder, suffix in (("AudioWAV", ".wav"), ("VideoFlash", ".flv")):
        for path in sorted((root / folder).glob(f"*{suffix}")):
            if path.name.startswith("._") or path.stem.split("_")[2] not in CLASSES:
                continue
            info = pointer(path)
            if info is None:
                already += 1
            else:
                jobs.append((f"{folder}/{path.name}", *info))
    print(f"to fetch {len(jobs)}, already media {already}", flush=True)
    results, done = {}, 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(fetch, root, rel, oid, size): rel for rel, oid, size in jobs}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
            done += 1
            if done % 250 == 0 or done == len(jobs):
                failed = sum(value.startswith("failed") for value in results.values())
                print(f"{done}/{len(jobs)} done, {failed} failed", flush=True)
    failures = {key: value for key, value in results.items() if value.startswith("failed")}
    Path(args.report).write_text(json.dumps({"requested": len(jobs), "already_media": already,
        "fetched": len(results) - len(failures), "failures": failures}, indent=1) + "\n")
    print("DONE", len(results) - len(failures), "fetched,", len(failures), "failed", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
