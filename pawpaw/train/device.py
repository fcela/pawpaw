"""Device + dtype selection. Avoids CUDA-only deps. cuda → mps → cpu."""
from __future__ import annotations

import os

import torch


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _cpu_supports_bf16() -> bool:
    """Check if the CPU has native BF16 support (AVX512_BF16 or AMX)."""
    if os.environ.get("PAWPAW_CPU_BF16") == "1":
        return True
    if os.environ.get("PAWPAW_CPU_BF16") == "0":
        return False
    try:
        if hasattr(torch, "cpu") and hasattr(torch.cpu, "is_bf16_supported"):
            return bool(torch.cpu.is_bf16_supported())
    except (AttributeError, RuntimeError):
        pass
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.optional.armv8_bf16"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip() == "1":
            return True
    except (OSError, subprocess.TimeoutExpired):
        pass
    return False


def pick_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    if device == "cpu" and _cpu_supports_bf16():
        return torch.bfloat16
    return torch.float32
