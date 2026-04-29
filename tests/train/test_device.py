from __future__ import annotations

import pytest

from pawpaw.train.device import pick_device, pick_dtype


def test_pick_device_falls_back_to_cpu(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False, raising=False)
    assert pick_device() == "cpu"


def test_pick_device_prefers_cuda(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True, raising=False)
    assert pick_device() == "cuda"


def test_pick_device_picks_mps_when_no_cuda(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True, raising=False)
    assert pick_device() == "mps"


@pytest.mark.parametrize("device,expected", [("cuda", "bfloat16"), ("mps", "float16"), ("cpu", "float32")])
def test_pick_dtype(device, expected):
    import torch
    import os
    monkeypatch_val = os.environ.get("PAWPAW_CPU_BF16")
    if device == "cpu" and monkeypatch_val:
        expected = "bfloat16" if monkeypatch_val == "1" else "float32"
    dtype = pick_dtype(device)
    assert dtype == getattr(torch, expected)
