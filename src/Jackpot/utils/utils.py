"""Reproducibility and sparsity accounting for popup-wrapped models."""

import random

import numpy as np
import torch
import torch.nn as nn

from src.Jackpot.pruning.popup import PoppedUpLayer


def get_effective_sparsity_info(model):
    """Fraction of weights set to zero after applying popup masks (popup layers only)."""
    total_zeros = 0
    total_count = 0

    for module in model.modules():
        if isinstance(module, PoppedUpLayer) and isinstance(module.module, (nn.Linear, nn.Conv2d)):
            effective_weight = module._masked_parameters()["weight"]
            total_zeros += (effective_weight == 0).sum().item()
            total_count += effective_weight.numel()

    return {"sparsity": total_zeros / total_count if total_count > 0 else 0.0,
            "zero_count":total_count,
            "total_count": total_count}

def set_seed(seed=0):
    """Fix Python, NumPy, and PyTorch RNGs (CUDA included if available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)