import copy
import torch.nn as nn
from models.layer import PoppedUpLayer

def is_popupifiable(module: nn.Module) -> bool:
    # only want to popupify these, ie ignore batchnorm
    return isinstance(module, (nn.Linear, nn.Conv2d))

def popupify_inplace(module: nn.Module, k=.5):
    for name, child in list(module.named_children()):
        # recurse first so we reach leaves inside bigger blocks
        popupify_inplace(child, k)

        # only replace Conv2d and Linear layers
        if is_popupifiable(child):
            setattr(module, name, PoppedUpLayer(child, k))

    return module

def popupify(network: nn.Module, k=.5):
    net_copy = copy.deepcopy(network)
    return popupify_inplace(net_copy, k)

def set_subnetwork_training_mode(model):
    for m in model.modules():
        if isinstance(m, PoppedUpLayer):
            m.set_subnetwork_training_mode()