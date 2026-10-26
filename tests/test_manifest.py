from pathlib import Path

import pandas as pd

from iemocap_mer.manifest import integrity_report
from iemocap_mer.splits import create_session_splits, validate_splits


def test_integrity_report_counts() -> None:
    frame = pd.DataFrame(
        {
            "utterance_id": ["a", "b"],
            "dialogue_id": ["d", "d"],
            "session": [1, 1],
            "speaker_id": ["Ses01F", "Ses01M"],
            "emotion_code": ["ang", "neu"],
            "emotion4": ["angry", "neutral"],
            "audio_exists": [True, True],
            "video_exists": [True, True],
            "transcript_exists": [True, True],
            "valid": [True, True],
            "distress_proxy": [True, False],
            "duration_seconds": [1.0, 2.0],
            "valence": [2.0, 3.0],
            "activation": [4.0, 3.0],
            "dominance": [3.0, 3.0],
        }
    )
    report = integrity_report(frame)
    assert report["utterances"] == 2
    assert report["emotion4_utterances"] == 2
    assert report["distress_proxy_count"] == 1


def test_session_splits_are_disjoint() -> None:
    rows = []
    for session in range(1, 6):
        for gender in "FM":
            rows.append(
                {
                    "utterance_id": f"Ses{session:02d}{gender}_x",
                    "session": session,
                    "speaker_id": f"Ses{session:02d}{gender}",
                }
            )
    frame = pd.DataFrame(rows)
    folds = create_session_splits(frame)
    validate_splits(frame, folds)
    assert len(folds) == 5
