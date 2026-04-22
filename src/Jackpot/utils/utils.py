from src.Jackpot.pruning.popup import PoppedUpLayer
import torch.nn as nn
import torch
import random
import numpy as np


def get_effective_sparsity_info(model):
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)