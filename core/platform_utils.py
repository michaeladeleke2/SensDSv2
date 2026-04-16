"""
core/platform_utils.py
Cross-platform helpers for SensDSv2.

Centralises device detection so every worker (train, inference, vex-aim) uses
the same logic instead of duplicating cuda/mps/cpu checks everywhere.
"""


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
