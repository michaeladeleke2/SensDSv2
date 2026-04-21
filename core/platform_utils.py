"""
core/platform_utils.py
Cross-platform helpers for SensDSv2.

Centralises device detection so every worker (train, inference, vex-aim) uses
the same logic instead of duplicating cuda/mps/cpu checks everywhere.
"""

import sys


def get_device():
    """
    Return the best available torch.device for the current machine.

    Priority:  CUDA (NVIDIA GPU)  >  MPS (Apple Silicon)  >  CPU
    On a Surface Pro 9 (Intel i7 + Iris Xe) this always returns CPU,
    which is correct — the CPU-only torch wheel is installed there.

    Returns None if torch is not importable (e.g. at import time before
    setup_windows.bat has been run).
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    except ImportError:
        return None


def device_label() -> str:
    """Human-readable device name for status messages."""
    dev = get_device()
    if dev is None:
        return "unknown (torch not loaded)"
    name = str(dev)
    if name == "cuda":
        try:
            import torch
            return f"cuda ({torch.cuda.get_device_name(0)})"
        except Exception:
            return "cuda"
    return name


def min_infer_gap_s() -> float:
    """
    Minimum seconds to wait after one inference completes before the next
    one starts in continuous modes (RoboSoccer, Maze Game).

    Purpose: prevents the inference thread from running back-to-back and
    continuously pegging the CPU, which starves the Qt game-animation timers
    and causes stuttering on low-power devices.

    Mac (MPS / Apple Silicon): inference is ~10× faster than CPU-only, so
    a short gap keeps the game snappy.

    Windows / CPU-only (Surface Pro 9, Intel): each inference call takes
    1–3 s; the longer gap gives the CPU breathing room between calls so
    the game timer and display thread stay smooth.

    CUDA users get a short gap — GPU inference is fast and runs independently.
    """
    dev = get_device()
    if dev is not None and str(dev) in ("cuda", "mps"):
        return 0.3   # hardware-accelerated — inference is fast
    # CPU-only path: Windows Surface or any CPU-only machine
    return 1.5
