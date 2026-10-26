from __future__ import annotations
import json
import zipfile
from pathlib import Path
from .common import *

def run_smoke(args):
    """Exercise actual project architectures with synthetic finite tensors."""
    import torch
    repo_import(args.repo)
    from iemocap_mer.training import build_model, forward_model, resolve_device
    from .export import alter_batch
    path = Path(args.out).resolve()
    if path.exists():
        raise FileExistsError("Choose a new smoke-test report path")
    device = resolve_device(args.device)
    torch.manual_seed(20260926)
    config = dict(hidden_dim=16, num_classes=4, dropout=0., num_heads=4, fusion_layers=1)
    batch = {"audio": torch.randn(2, 16, 6, device=device),
        "video": torch.randn(2, 5, 5, device=device),
        "audio_mask": torch.ones(2, 16, dtype=torch.bool, device=device),
        "video_mask": torch.ones(2, 5, dtype=torch.bool, device=device),
        "utterance_id": ["test-a", "test-b"]}
    batch["audio_mask"][1, 12:] = False
    batch["video_mask"][1, 3:] = False
    checks = []
    try:
        with torch.inference_mode():
            for name in ("audio", "concat_matched", "essence"):
                model = build_model(name, 6, 5, config).to(device).eval()
                for condition in applicable_conditions(name):
                    changed = alter_batch(batch, condition, 20260926)
                    output = forward_model(model, name, changed)["logits"]
                    if output.shape != (2, 4) or not bool(torch.isfinite(output).all()):
                        raise AssertionError(f"{name}/{condition}: non-finite or wrong-shaped logits")
                    if condition.startswith("branch_disabled_"):
                        modality = condition.removeprefix("branch_disabled_")
                        other = dict(changed)
                        other[modality] = torch.randn_like(other[modality]) * 100
                        repeated = forward_model(model, name, other)["logits"]
                        if not torch.allclose(output, repeated, atol=1e-6, rtol=1e-6):
                            raise AssertionError(f"{name}/{condition}: disabled input changed logits")
                    checks.append({"model": name, "condition": condition, "passed": True})
        report = {"status": "passed", "device": str(device), "torch": torch.__version__,
            "checks": checks, "checkpoint_training_tested": False,
            "fully_masked_paths_tested": False, "real_checkpoint_export_tested": False}
        write_json(path, report)
        print(f"Project model smoke tests passed: {len(checks)} conditions")
        return 0
    except Exception as exc:
        write_json(path, {"status": "failed", "checks": checks, "error": str(exc)})
        raise

def run_ravdess_manifest(args):
    """Prepare the proposed four-class full-AV speech inventory, not a trainer."""
    root = Path(args.raw).resolve()
    out = new_output(args.out)
    label_map = {"05": 0, "03": 1, "01": 2, "04": 3}
    rows, seen = [], set()
    for path in sorted(root.rglob("*.mp4")):
        parts = path.stem.split("-")
        if len(parts) != 7 or not all(p.isdigit() and len(p) == 2 for p in parts):
            continue
        modality, channel, emotion, intensity, statement, repetition, actor = parts
        if modality != "01" or channel != "01" or emotion not in label_map:
            continue
        if not (1 <= int(actor) <= 24 and intensity in ("01", "02") and
                statement in ("01", "02") and repetition in ("01", "02")):
            raise ValueError("Unexpected RAVDESS filename fields")
        if emotion == "01" and intensity != "01":
            raise ValueError("Unexpected strong neutral clip")
        if path.stem in seen:
            raise ValueError("Duplicate RAVDESS stimulus: " + path.stem)
        seen.add(path.stem)
        rows.append(dict(dataset="ravdess", utterance_id=path.stem, speaker_id=int(actor),
            label=label_map[emotion], emotion=("angry", "happy", "neutral", "sad")[label_map[emotion]],
            relative_video_path=str(path.relative_to(root)), file_sha256=sha256(path)))
    if len(rows) != 672 or any(sum(r["speaker_id"] == a for r in rows) != 28 for a in range(1, 25)):
        raise ValueError(f"Expected 672 selected AV speech clips (28 per actor), found {len(rows)}")
    groups = [set(range(1, 6)), set(range(6, 11)), set(range(11, 16)), set(range(16, 21)), set(range(21, 25))]
    split_rows = []
    for f in range(5):
        for row in rows:
            actor = row["speaker_id"]
            part = "test" if actor in groups[f] else "validation" if actor in groups[(f + 1) % 5] else "train"
            split_rows.append({k: row[k] for k in ("dataset", "utterance_id", "speaker_id", "label")} |
                              {"fold": f + 1, "split": part})
    write_csv(out / "ravdess_manifest.csv", rows)
    write_csv(out / "split_manifest.csv", split_rows)
    write_json(out / "protocol.json", {"created": timestamp(), "status": "manifest_only_not_trained",
        "source": "https://zenodo.org/records/1188976", "test_actor_groups": [sorted(g) for g in groups],
        "development_rule": "next group cyclically", "labels": ["angry", "happy", "neutral", "sad"],
        "seeds": list(SEEDS), "planned_training_runs": 75,
        "remaining": ["Verify AV decoding and alignment; file hashes do not verify media contents.",
            "Implement a RAVDESS feature-path/data adapter and extract all three representations.",
            "Skip unavailable VA targets and VA loss/metrics in the actual trainer.",
            "Lock the full protocol before held-out evaluation, then train/evaluate the 75-run grid."]})
    print("RAVDESS inventory and proposed splits saved; no features or models were generated.")
    return 0

def run_pack(args):
    root = Path(args.outputs).resolve()
    if not root.is_dir():
        raise ValueError("--outputs must be the results folder")
    output = Path(args.out).resolve()
    if output.exists():
        raise FileExistsError("Refusing to replace an existing results ZIP")
    output.parent.mkdir(parents=True, exist_ok=True)
    extensions = {".csv", ".json", ".jsonl", ".txt", ".log", ".md", ".yaml", ".yml", ".png",
                  ".pdf", ".tex", ".bib", ".svg", ".py", ".sh", ".toml", ".patch"}
    selected, omitted = [], []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if path.is_symlink() or any(p.startswith(".") or p == "__pycache__" for p in relative.parts):
            omitted.append(str(relative)); continue
        if path.suffix.lower() not in extensions and not path.name.endswith(".csv.gz"):
            omitted.append(str(relative)); continue
        selected.append(path)
    code_root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            archive.write(path, "mtap_outputs/" + str(path.relative_to(root)))
        for path in sorted(code_root.rglob("*")):
            if path.is_file() and path.suffix in (".py", ".md", ".txt", ".json") and "__pycache__" not in path.parts:
                archive.write(path, "companion_code/" + str(path.relative_to(code_root)))
        archive.writestr("PACKING_REPORT.json", json.dumps({"files": len(selected), "omitted": omitted,
            "created": timestamp(), "warning": "Run on mtap_outputs only; inspect logs before sharing."}, indent=2))
    print(f"Created {output} ({output.stat().st_size} bytes)")
    return 0
