from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_seeds(seed: Optional[int]) -> int:
    """Seed Python, NumPy, Torch. If seed is None or <0, derive one."""
    if seed is None or seed < 0:
        seed = int.from_bytes(os.urandom(4), "big") % (2**31 - 1)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior where feasible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

