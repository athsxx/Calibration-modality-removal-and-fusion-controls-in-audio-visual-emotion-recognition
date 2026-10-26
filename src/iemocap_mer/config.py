from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    base = path.parent.parent
    for key in ("dataset_root", "artifacts_root"):
        value = Path(config["paths"][key]).expanduser()
        if not value.is_absolute():
            value = (base / value).resolve()
        config["paths"][key] = str(value)
    paths = config.setdefault("paths", {})
    paths.setdefault("manifest_file", "manifests/iemocap_utterances.csv")
    paths.setdefault("splits_file", "splits/emotion4_session_folds.json")
    data = config.setdefault("data", {})
    data.setdefault("has_va", True)
    data.setdefault("dataset_name", "iemocap")
    features = config.setdefault("features", {})
    features.setdefault("face_region", "speaker_half")
    training = config.setdefault("training", {})
    # Off unless a config sets it. Existing IEMOCAP yaml files do not.
    training.setdefault("cml_weight", 0.0)
    # "penalty" is the existing hinge. "faithful" is Algorithm 1. Absent means penalty.
    training.setdefault("cml_variant", "penalty")
    # On unless a config sets it false. Existing yaml weights are unchanged.
    training.setdefault("directional_distillation", True)
    return config

