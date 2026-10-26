from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .manifest import build_manifest, integrity_report, write_manifest
from .splits import write_splits
from .features import (
    LogMelExtractor,
    VideoExtractor,
    Wav2VecExtractor,
    extract_features,
    extract_video_features_batched,
    verify_feature_cache,
)
from .features_au import VideoAUExtractor, extract_video_au_features_batched
from .corpora import CorpusCountError, build_corpus_manifest
from .training import CheckpointExistsError, NonFiniteTrainingAbort, Stage2NotReached, run_training
from .analysis import distress_proxy_analysis, run_posthoc
from .reporting import aggregate_runs, audit_video_quality, generate_research_report


def _paths(config: dict) -> tuple[Path, Path]:
    return Path(config["paths"]["dataset_root"]), Path(config["paths"]["artifacts_root"])


def _under_artifacts(artifacts: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else artifacts / path


def _manifest_path(config: dict, artifacts: Path) -> Path:
    return _under_artifacts(artifacts, config["paths"]["manifest_file"])


def _splits_path(config: dict, artifacts: Path) -> Path:
    return _under_artifacts(artifacts, config["paths"]["splits_file"])


def command_manifest(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    root, artifacts = _paths(config)
    proxy = config["data"]["distress_proxy"]
    frame = build_manifest(root, proxy["valence_lt"], proxy["activation_gt"])
    output = _manifest_path(config, artifacts)
    report = artifacts / "reports" / "integrity_report.json"
    write_manifest(frame, output, report)
    print(json.dumps(integrity_report(frame), indent=2))


def command_splits(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _, artifacts = _paths(config)
    frame = pd.read_csv(_manifest_path(config, artifacts))
    if args.task == "emotion4":
        frame = frame[frame.emotion4.notna() & frame.emotion4.ne("")].copy()
    folds = write_splits(frame, _splits_path(config, artifacts))
    summary = {
        fold: {part: len(values[part]) for part in ("train", "dev", "test")}
        for fold, values in folds.items()
    }
    print(json.dumps(summary, indent=2))


def command_extract(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _, artifacts = _paths(config)
    frame = pd.read_csv(_manifest_path(config, artifacts))
    if args.task == "emotion4":
        frame = frame[frame.emotion4.notna() & frame.emotion4.ne("")].copy()
    if args.session is not None:
        frame = frame[frame.session.eq(args.session)].copy()
    if args.limit is not None:
        frame = frame.head(args.limit)
    device = args.device
    if args.modality == "audio":
        if args.backend == "logmel":
            extractor = LogMelExtractor(
                config["features"]["audio_sample_rate"], config["features"]["n_mels"]
            )
        else:
            extractor = Wav2VecExtractor(device=device)
    else:
        video_backend = str(config["features"].get("video_backend", "resnet18"))
        face_region = str(config["features"].get("face_region", "speaker_half"))
        if args.backend == "openface_au" or video_backend == "openface_au":
            extractor = VideoAUExtractor(
                fps=config["features"]["video_fps"],
                max_frames=config["features"]["video_max_frames"],
                face_margin=float(config["features"].get("face_margin", 0.25)),
                device=device,
                batch_size=int(config["features"].get("au_batch_size", 16)),
                face_region=face_region,
            )
        else:
            extractor = VideoExtractor(
                fps=config["features"]["video_fps"],
                max_frames=config["features"]["video_max_frames"],
                pretrained=args.pretrained,
                device=device,
                face_margin=float(config["features"].get("face_margin", 0.25)),
                face_region=face_region,
            )
    video_dir = str(config["features"].get("video_dir", "video"))
    audio_dir = str(config["features"].get("audio_dir", "audio"))
    if args.modality == "video":
        feature_dir = video_dir
    elif args.modality == "audio":
        feature_dir = audio_dir
    else:
        feature_dir = None
    if args.modality == "video":
        if isinstance(extractor, VideoAUExtractor):
            report = extract_video_au_features_batched(
                frame,
                artifacts / "features",
                extractor,
                overwrite=args.overwrite,
                video_dir=video_dir,
            )
        else:
            report = extract_video_features_batched(
                frame,
                artifacts / "features",
                extractor,
                overwrite=args.overwrite,
                encoder_batch_size=args.encoder_batch_size,
                video_dir=video_dir,
            )
    else:
        report = extract_features(
            frame,
            artifacts / "features",
            args.modality,
            extractor,
            overwrite=args.overwrite,
            feature_dir=feature_dir,
        )
    print(json.dumps(report, indent=2))


def command_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _, artifacts = _paths(config)
    frame = pd.read_csv(_manifest_path(config, artifacts))
    frame = frame[frame.emotion4.notna() & frame.emotion4.ne("")].copy()
    folds = json.loads(_splits_path(config, artifacts).read_text(encoding="utf-8"))
    for loss_name in args.disable_loss:
        config["training"]["loss_weights"][loss_name] = 0.0
    if args.device:
        config["training"]["device"] = args.device
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.modality_dropout_audio is not None or args.modality_dropout_video is not None:
        modality_dropout = config["training"].setdefault("modality_dropout", {})
        if args.modality_dropout_audio is not None:
            modality_dropout["audio"] = args.modality_dropout_audio
        if args.modality_dropout_video is not None:
            modality_dropout["video"] = args.modality_dropout_video
    suffix = f"_{args.tag}" if args.tag else ""
    checkpoint = artifacts / "checkpoints" / f"{args.model}_{args.fold}{suffix}.pt"
    try:
        result = run_training(
            frame, folds[args.fold], artifacts / "features", args.model, config, checkpoint
        )
    except (NonFiniteTrainingAbort, Stage2NotReached, CheckpointExistsError) as exc:
        if not isinstance(exc, NonFiniteTrainingAbort):
            print(json.dumps({"error": type(exc).__name__, "message": str(exc)}))
        raise SystemExit(1) from None
    print(json.dumps(result["test"], indent=2))


def command_verify_features(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _, artifacts = _paths(config)
    frame = pd.read_csv(_manifest_path(config, artifacts))
    frame = frame[frame.emotion4.notna() & frame.emotion4.ne("")].copy()
    video_dir = str(config["features"].get("video_dir", "video")) if args.modality == "video" else None
    audio_dir = str(config["features"].get("audio_dir", "audio")) if args.modality == "audio" else None
    feature_dir = video_dir or audio_dir
    report = verify_feature_cache(
        frame, artifacts / "features", args.modality, feature_dir=feature_dir
    )
    print(json.dumps({key: value for key, value in report.items() if key not in ("missing", "invalid")}, indent=2))


def command_analyze(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _, artifacts = _paths(config)
    frame = pd.read_csv(_manifest_path(config, artifacts))
    emotion4 = frame[frame.emotion4.notna() & frame.emotion4.ne("")].copy()
    folds = json.loads(_splits_path(config, artifacts).read_text(encoding="utf-8"))
    suffix = f"_{args.tag}" if args.tag else ""
    checkpoint = artifacts / "checkpoints" / f"{args.model}_{args.fold}{suffix}.pt"
    robustness = run_posthoc(
        checkpoint,
        emotion4,
        folds[args.fold],
        artifacts / "features",
        artifacts / "results",
        config["training"]["device"],
        config["training"]["batch_size"],
        video_dir=str(config["features"].get("video_dir", "video")),
        audio_dir=str(config["features"].get("audio_dir", "audio")),
        video_max_tokens=int(config["features"]["video_max_frames"]),
        has_va=bool(config["data"].get("has_va", True)),
    )
    predictions = pd.read_csv(checkpoint.with_suffix(".predictions.csv"))
    distress = distress_proxy_analysis(predictions, frame)
    distress_path = artifacts / "results" / f"{checkpoint.stem}_distress_proxy.json"
    distress_path.write_text(json.dumps(distress, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"robustness": robustness, "distress_proxy": distress}, indent=2))


def command_video_audit(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _, artifacts = _paths(config)
    frame = audit_video_quality(
        artifacts,
        _manifest_path(config, artifacts),
    )
    output = artifacts / "reports" / "video_quality_audit.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))


def command_research_report(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _, artifacts = _paths(config)
    report = generate_research_report(
        artifacts / "checkpoints",
        artifacts / "results",
        artifacts / "reports",
    )
    print(json.dumps({"models": len(report["model_ranking"]), "conclusions": report["conclusions"]}, indent=2))


def command_build_corpus_manifest(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _, artifacts = _paths(config)
    try:
        report = build_corpus_manifest(
            args.corpus,
            args.raw,
            artifacts,
            _manifest_path(config, artifacts),
        )
    except CorpusCountError as exc:
        print(json.dumps({"error": "corpus_count", "message": str(exc)}))
        raise SystemExit(1) from None
    print(json.dumps({key: value for key, value in report.items() if key != "decode_failures"}, indent=2))
    if report["decode_failures"]:
        print(json.dumps({"decode_failure_count": report["decode_failure_count"]}))
        raise SystemExit(1)


def command_summarize(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    _, artifacts = _paths(config)
    runs = aggregate_runs(
        artifacts / "checkpoints", artifacts / "results" / "cross_fold_metrics.csv"
    )
    print(f"Aggregated {len(runs)} completed runs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="iemocap-mer")
    parser.add_argument("--config", default="configs/default.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("build-manifest")
    manifest.set_defaults(function=command_manifest)
    corpus_manifest = commands.add_parser("build-corpus-manifest")
    corpus_manifest.add_argument("--corpus", choices=("ravdess", "cremad"), required=True)
    corpus_manifest.add_argument("--raw", required=True, help="Directory of raw corpus files")
    corpus_manifest.set_defaults(function=command_build_corpus_manifest)
    splits = commands.add_parser("build-splits")
    splits.add_argument("--task", choices=("emotion4", "all"), default="emotion4")
    splits.set_defaults(function=command_splits)
    extract = commands.add_parser("extract-features")
    extract.add_argument("--modality", choices=("audio", "video"), required=True)
    extract.add_argument(
        "--backend",
        choices=("logmel", "wav2vec2", "openface_au"),
        default="logmel",
        help="Audio backend or openface_au for py-feat AU video features",
    )
    extract.add_argument("--task", choices=("emotion4", "all"), default="emotion4")
    extract.add_argument("--limit", type=int)
    extract.add_argument("--session", type=int, help="Restrict extraction to one IEMOCAP session")
    extract.add_argument("--device", default="cpu")
    extract.add_argument("--pretrained", action="store_true")
    extract.add_argument("--overwrite", action="store_true")
    extract.add_argument("--encoder-batch-size", type=int, default=64)
    extract.set_defaults(function=command_extract)
    verify = commands.add_parser("verify-features")
    verify.add_argument("--modality", choices=("audio", "video"), required=True)
    verify.set_defaults(function=command_verify_features)
    train = commands.add_parser("train")
    train.add_argument(
        "--model",
        choices=("audio", "video", "concat", "concat_matched", "cross_attention", "essence"),
        required=True,
    )
    train.add_argument("--fold", choices=tuple(f"fold_{i}" for i in range(1, 6)), required=True)
    train.add_argument("--device", help="Override training device (cpu, mps, cuda)")
    train.add_argument("--tag", help="Checkpoint suffix for a named ablation")
    train.add_argument("--seed", type=int, help="Override training.seed from the config (for multi-seed runs)")
    train.add_argument(
        "--disable-loss",
        action="append",
        default=[],
        choices=(
            "va",
            "alignment",
            "orthogonality",
            "audio_to_video_distillation",
            "video_to_audio_distillation",
        ),
    )
    train.add_argument(
        "--modality-dropout-audio",
        type=float,
        help="Train-time probability of zero-filling audio per sample (eval unchanged)",
    )
    train.add_argument(
        "--modality-dropout-video",
        type=float,
        help="Train-time probability of zero-filling video per sample (eval unchanged)",
    )
    train.set_defaults(function=command_train)
    analyze = commands.add_parser("analyze")
    analyze.add_argument(
        "--model",
        choices=("audio", "video", "concat", "concat_matched", "cross_attention", "essence"),
        required=True,
    )
    analyze.add_argument("--fold", choices=tuple(f"fold_{i}" for i in range(1, 6)), required=True)
    analyze.add_argument("--tag", help="Checkpoint suffix for a named ablation")
    analyze.set_defaults(function=command_analyze)
    summarize = commands.add_parser("summarize")
    summarize.set_defaults(function=command_summarize)
    video_audit = commands.add_parser("video-audit")
    video_audit.set_defaults(function=command_video_audit)
    research_report = commands.add_parser("research-report")
    research_report.set_defaults(function=command_research_report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
