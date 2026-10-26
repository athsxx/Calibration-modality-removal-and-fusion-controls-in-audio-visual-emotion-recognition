# Calibration, modality removal, and fusion controls in audio–visual emotion recognition

Code, locked analysis protocols, and derived results for the three-corpus study of the same name by Atharv Jaju and Ravindra Kumar Singh (CHRIST (Deemed to be University), Bengaluru).

The study asks whether confidence in audio–visual emotion recognition stays reliable when a modality is removed. It evaluates IEMOCAP, RAVDESS, and CREMA-D with speaker-disjoint five-fold splits and three seeds (0, 1, 7), and compares:

- fusion architectures (matched concatenation and Essence) with two visual representations (facial action units and ResNet-18),
- development-only temperature scaling,
- a confidence-ranking penalty inspired by calibrated multimodal learning (CML) and an implementation of the published two-modality CML objective,
- distillation-off and parameter-budget controls, and selective fusion.

Outcomes are reported for clean inputs and two removal operators (zero-fill and branch-disabled). Speaker-cluster bootstrap intervals and Holm-adjusted fold tests classify each contrast as different, equivalent (within ±0.02 macro-F1), or inconclusive.

## What is in this repository

| Path | Contents |
|---|---|
| `src/iemocap_mer/` | Package: manifests, splits, feature extraction, models, losses (including both CML variants), training, metrics, speaker-level statistics |
| `configs/` | One YAML per corpus and model variant used in the study |
| `scripts/` | Training grids, prediction export, combined comparison views, statistical analysis, figures |
| `mer_companion/` | Export and analysis companion (`merkit`) used by `scripts/mtap_export_corpus.py` |
| `tests/`, `mer_companion/tests/` | Unit and regression tests |
| `mtap_outputs/protocol/` | Locked analysis protocol (v1) and dated amendments (v2–v5) with SHA-256 files, plus the pre-registered comparison pairs |
| `mtap_outputs/analysis*/` | Speaker-level analysis tables for every reported comparison (see below) |
| `mtap_outputs/selective_v3_*/` | Selective-fusion decision tables and analyses |
| `mtap_outputs/export_manifests/` | Run inventories for every prediction export (run ids, checkpoint and metrics hashes, eligibility) |
| `mtap_outputs/*.md` | Result notes (faithful CML, CML fidelity audit, Step 9 controls, selective fusion) and `DATA_DICTIONARY.md` |
| `artifacts*/splits/` | Speaker-disjoint fold assignments (utterance ids only) for each corpus |

### Analysis folders

| Folder | Comparison |
|---|---|
| `analysis_strong_v2` | IEMOCAP principal grid (post hoc family) |
| `analysis_strong_ravdess`, `analysis_strong_cremad` | RAVDESS and CREMA-D principal grids |
| `analysis_cml_<corpus>` | CML-inspired penalty versus ConcatMatched-AU |
| `analysis_cmlf_<corpus>` | Faithful CML versus ConcatMatched-AU and versus the penalty |
| `analysis_step9_<corpus>` | Distillation-off and parameter-budget controls |
| `analysis_strong`, `analysis` | Earlier IEMOCAP passes, retained for the record and not used for claims |

Each folder has `paired_comparisons.csv` (effects, intervals, consensus verdicts), `point_and_ci_metrics.csv` (per-system estimates, raw and temperature-scaled), per-class, variance-component, sensitivity and minimum-detectable-effect tables, and `strong_analysis_manifest.json`. Column definitions are in `mtap_outputs/DATA_DICTIONARY.md`.

### Not included

The corpora, transcripts, per-utterance labels, extracted features, model weights, and per-utterance predictions are not distributed. Machine-specific paths in manifests are replaced by `<REPO>` (repository root) and `<DATA_ROOT>` (data volume).

## Data

Obtain each corpus from its provider under its own terms:

- **IEMOCAP**: USC SAIL release agreement, <https://sail.usc.edu/iemocap/>
- **RAVDESS**: CC BY-NC-SA 4.0, <https://zenodo.org/records/1188976>
- **CREMA-D**: ODbL v1.0 / DbCL v1.0, <https://github.com/CheyneyComputerScience/CREMA-D>

The configs expect the corpora next to the repository (`paths.dataset_root: ".."`). IEMOCAP's `Session1`–`Session5` folders sit in that parent directory; RAVDESS and CREMA-D manifests are built from a raw-file directory passed with `--raw`.

## Installation

Python 3.10 or later. The study used Python 3.11.8, PyTorch 2.13.0, torchaudio 2.11.0, torchvision 0.28.0, and py-feat 2.1.1 on macOS (Apple MPS).

```bash
python -m pip install -e ".[plots,test]"
python -m pip install -e ".[au]"   # only needed to extract facial action units
```

`ffmpeg` and `ffprobe` must be on `PATH` (or set `FFMPEG` / `FFPROBE`) to build RAVDESS and CREMA-D manifests.

## Quick checks without the corpora

```bash
python -m pytest                       # 92 tests; one checkpoint test is skipped without weights
python scripts/mtap_make_figures.py    # redraws the figures from mtap_outputs/analysis* into mtap_outputs/figures
```

To recompute a speaker-level analysis from an existing prediction export:

```bash
python scripts/mtap_strong_analysis.py --exports <export_dir> --out <analysis_dir>
```

## Reproducing the experiments

All commands run from the repository root with `export PYTHONPATH=src`. The shell scripts use `$PY` (default `python3`) and are resumable: completed runs are skipped and failed runs are kept.

1. **Manifests and splits.** `python -m iemocap_mer.cli build-manifest` and `build-splits --task emotion4` for IEMOCAP; `build-corpus-manifest --corpus {ravdess,cremad} --raw <dir>` with the corpus config for the others. The committed `artifacts*/splits/` files are the folds used in the study.
2. **Features.** `extract-features --modality audio --backend wav2vec2` and the action-unit extraction (`scripts/run_video_au_extract.sh` for IEMOCAP; `scripts/mtap_overnight.sh` for RAVDESS and CREMA-D).
3. **Training grids.** `scripts/run_multiseed_grid.sh` (IEMOCAP principal grid), `scripts/mtap_overnight.sh` (RAVDESS and CREMA-D grids), `scripts/mtap_cml_grid.sh` (CML-inspired penalty), `scripts/mtap_cml_faithful_grid.sh` (faithful CML), `scripts/mtap_step9_grid.sh` (distillation-off and parameter-budget controls). The comparison grids stage exports in `$STAGING` (default `../<name>_export_staging`).
4. **Export and analysis.** `scripts/mtap_export_corpus.py` writes identified logits for every removal condition; `scripts/build_cml_combined_view.py` and `scripts/build_step9_combined_view.py` assemble comparison views; `scripts/mtap_strong_analysis.py` produces the analysis folders; `scripts/mtap_selective_pilot.py` runs selective fusion.
5. **Figures.** `scripts/mtap_make_figures.py`.

The protocol files in `mtap_outputs/protocol/` record which comparisons, margins and decision rules were fixed before each set of test scores was inspected. Historical IEMOCAP analyses are post hoc; RAVDESS, CREMA-D, CML, faithful CML and Step 9 follow the dated amendments.

## Provenance

This repository is a cleaned export of the development repository at commit `bdc2224241d1d7d1641aa8947d47152c43fde39c`, the experimental snapshot cited in the manuscript. Code and configs are unchanged except for portability: interpreter, staging and corpus paths in scripts became overridable defaults, `ffmpeg`/`ffprobe` are read from `PATH`, one test reads the running interpreter, and the checkpoint-loading test skips when weights are absent.

## Licence

Code is released under the Apache License 2.0 (`LICENSE`). The corpora remain under their own licences.

## Citation

See `CITATION.cff`. Publication details will be added when available.
