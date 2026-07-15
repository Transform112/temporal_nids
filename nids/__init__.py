"""NIDS — GNN-based Network Intrusion Detection System.

Temporally-aware, self-supervised, adversarially-robust, explainable GNN for NIDS.
"""

import random
import numpy as np
import os

# ── Single source of truth for random seed ──────────────────────────────
SEED: int = 42

os.environ["PYTHONHASHSEED"] = str(SEED)


def set_seed(seed: int = SEED) -> None:
    """Set all random seeds for reproducibility. Call once at script entry."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ── Compute target detection ─────────────────────────────────────────────
def get_compute_target() -> str:
    """Detect whether we're on laptop or Kaggle. Returns 'laptop' or 'kaggle'."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    return "laptop"


COMPUTE_TARGET: str = get_compute_target()

__version__ = "0.1.0"
