from pathlib import Path

import pytest
import torch
from torch import nn

from iemocap_mer.training import (
    NonFiniteMonitor,
    NonFiniteTrainingAbort,
    classify_grad_nonfinite,
    classify_tensor_nonfinite,
)


def test_classify_tensor_nonfinite_nan_and_inf() -> None:
    nan_t = torch.tensor([1.0, float("nan")])
    inf_t = torch.tensor([1.0, float("inf")])
    ok_t = torch.tensor([1.0, 2.0])
    assert classify_tensor_nonfinite(nan_t) == (True, False)
    assert classify_tensor_nonfinite(inf_t) == (False, True)
    assert classify_tensor_nonfinite(ok_t) == (False, False)
    assert classify_tensor_nonfinite(None) == (False, False)


def test_nan_loss_is_recorded_and_stops_loop(tmp_path: Path) -> None:
    event_path = tmp_path / "run.nonfinite_events.jsonl"
    monitor = NonFiniteMonitor(event_path, max_batches=1)
    loss = torch.tensor(float("nan"))

    with pytest.raises(NonFiniteTrainingAbort) as raised:
        monitor.check_loss(loss, epoch=3, batch_index=7, stage=1)

    assert raised.value.count == 1
    text = event_path.read_text(encoding="utf-8")
    assert "loss_nan" in text
    assert '"epoch": 3' in text
    assert '"batch_index": 7' in text
    assert monitor.count == 1


def test_finite_loss_then_nan_grad_aborts(tmp_path: Path) -> None:
    event_path = tmp_path / "run.nonfinite_events.jsonl"
    monitor = NonFiniteMonitor(event_path, max_batches=1)
    model = nn.Linear(2, 2)
    model.weight.grad = torch.full_like(model.weight, float("nan"))
    model.bias.grad = torch.zeros_like(model.bias)

    assert classify_grad_nonfinite(model) == (True, False)
    with pytest.raises(NonFiniteTrainingAbort):
        monitor.check_grads(model, epoch=1, batch_index=0, stage=2)

    event = monitor.events[0]
    assert event["grad_nan"] is True
    assert event["where"] == "grad"


def test_threshold_allows_then_aborts(tmp_path: Path) -> None:
    monitor = NonFiniteMonitor(tmp_path / "events.jsonl", max_batches=2)
    inf_loss = torch.tensor(float("inf"))
    assert monitor.check_loss(inf_loss, epoch=1, batch_index=0, stage=1) is False
    with pytest.raises(NonFiniteTrainingAbort):
        monitor.check_loss(inf_loss, epoch=1, batch_index=1, stage=1)
    assert monitor.count == 2
