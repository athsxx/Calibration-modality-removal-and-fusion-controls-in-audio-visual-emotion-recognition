from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def create_session_splits(frame: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    """Create five speaker-disjoint train/dev/test folds.

    Test is the outer LOSO session. Dev is the following session cyclically;
    the other three sessions train the model. This protocol is deterministic
    and prevents actor leakage in model selection and final evaluation.
    """
    sessions = sorted(int(value) for value in frame.session.unique())
    if sessions != [1, 2, 3, 4, 5]:
        raise ValueError(f"Expected sessions 1..5, found {sessions}")

    folds: dict[str, dict[str, list[str]]] = {}
    for test_session in sessions:
        dev_session = sessions[test_session % len(sessions)]
        train_sessions = [
            session for session in sessions if session not in (test_session, dev_session)
        ]
        fold = f"fold_{test_session}"
        folds[fold] = {
            "train": frame.loc[frame.session.isin(train_sessions), "utterance_id"].tolist(),
            "dev": frame.loc[frame.session.eq(dev_session), "utterance_id"].tolist(),
            "test": frame.loc[frame.session.eq(test_session), "utterance_id"].tolist(),
            "train_sessions": train_sessions,
            "dev_sessions": [dev_session],
            "test_sessions": [test_session],
        }
    return folds


def validate_splits(frame: pd.DataFrame, folds: dict) -> None:
    lookup = frame.set_index("utterance_id")
    all_ids = set(frame.utterance_id)
    for name, fold in folds.items():
        sets = {part: set(fold[part]) for part in ("train", "dev", "test")}
        if (sets["train"] & sets["dev"]) or (sets["train"] & sets["test"]) or (
            sets["dev"] & sets["test"]
        ):
            raise ValueError(f"Utterance leakage in {name}")
        if set.union(*sets.values()) != all_ids:
            raise ValueError(f"Incomplete coverage in {name}")
        speaker_sets = {
            part: set(lookup.loc[list(ids), "speaker_id"].tolist())
            for part, ids in sets.items()
        }
        if any(
            speaker_sets[left] & speaker_sets[right]
            for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))
        ):
            raise ValueError(f"Speaker leakage in {name}")


def write_splits(frame: pd.DataFrame, output_path: str | Path) -> dict:
    folds = create_session_splits(frame)
    validate_splits(frame, folds)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")
    return folds

