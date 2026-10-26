#!/usr/bin/env python3
"""Inference latency for the locked Step 9 architectures.

Measures untrained forwards. Weights are not fit on development or test data.
Writes a new JSON file and refuses to overwrite one.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from iemocap_mer.models import ConcatFusionModelMatched, EssenceTransferModel


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def peak_bytes(device: torch.device) -> int | None:
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return int(torch.mps.current_allocated_memory())
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated())
    return None


def time_model(model, audio, audio_mask, video, video_mask, device, warmup: int, repeats: int) -> dict:
    model = model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(warmup):
            model(audio, audio_mask, video, video_mask)
        synchronize(device)
        started = time.perf_counter()
        for _ in range(repeats):
            model(audio, audio_mask, video, video_mask)
        synchronize(device)
        elapsed = time.perf_counter() - started
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "trainable_parameters": trainable,
        "seconds_per_batch": elapsed / repeats,
        "batches": repeats,
        "warmup": warmup,
        "peak_allocated_bytes": peak_bytes(device),
    }


def main(argv: list[str] | None = None) -> int:
    tool = argparse.ArgumentParser(description=__doc__)
    tool.add_argument("--out", required=True)
    tool.add_argument("--device", default="mps")
    tool.add_argument("--batch-size", type=int, default=16)
    tool.add_argument("--warmup", type=int, default=10)
    tool.add_argument("--repeats", type=int, default=50)
    args = tool.parse_args(argv)
    destination = Path(args.out)
    if destination.exists():
        print(f"Refusing to overwrite {destination}", file=sys.stderr)
        return 2
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    device = torch.device(args.device)
    batch = args.batch_size
    audio = torch.randn(batch, 400, 768, device=device)
    video = torch.randn(batch, 32, 20, device=device)
    audio_mask = torch.ones(batch, 400, dtype=torch.bool, device=device)
    video_mask = torch.ones(batch, 32, dtype=torch.bool, device=device)
    models = {
        "concat_matched_au": ConcatFusionModelMatched(768, 20, 256, 4, dropout=0.0),
        "essence_au": EssenceTransferModel(768, 20, 256, 4, num_heads=4, fusion_layers=2, dropout=0.0),
        "concat_matched_au_budget": ConcatFusionModelMatched(
            768, 20, 256, 4, dropout=0.0, fusion_width=2032, encoder_layers=1
        ),
    }
    report = {
        "device": args.device,
        "batch_size": batch,
        "audio_tokens": 400,
        "video_tokens": 32,
        "audio_dim": 768,
        "video_dim": 20,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "note": "Untrained weights. Dropout disabled so timing is the forward graph only.",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "models": {},
    }
    for name, model in models.items():
        report["models"][name] = time_model(
            model, audio, audio_mask, video, video_mask, device, args.warmup, args.repeats
        )
        print(name, report["models"][name]["seconds_per_batch"], flush=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
