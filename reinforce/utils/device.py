from __future__ import annotations

import torch


def get_torch_device() -> torch.device:
    """Return best-available torch device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # On Apple Silicon, MPS may be available if built with Metal
    mps_available = getattr(torch.backends, "mps", None)
    if mps_available and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")

