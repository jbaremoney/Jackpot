import torch
import torch.nn as nn
from torch import autograd
from torch.func import functional_call
import copy

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):

        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1-k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, grad):

        # send the gradient g straight-through on the backward pass.
        return grad, None

"""general class to wrap a module, 'popupify' the trainable parameters"""
class PoppedUpLayer(nn.Module):
    def __init__(self, module: nn.Module, k: float = 0.5, just_weight: bool = True):
        super().__init__()
        self.module = module
        self.k = k
        self.just_weight = just_weight

        # Freeze original module params by default
        for p in self.module.parameters():
            p.requires_grad_(False)

        self.popup_scores = nn.ParameterDict()

        for name, param in module.named_parameters():
            # If just_weight=True, only create popup scores for the weight param
            if self.just_weight and name != "weight":
                continue

            score = nn.Parameter(torch.randn(2 * param.shape[0], *param.shape[1:])).to(param.device)
            safe_name = name.replace(".", "__")
            self.popup_scores[safe_name] = score

    def _score_key(self, name: str) -> str:
        return name.replace(".", "__")

    # ----- using this stuff for training after pruned ---
    def freeze_popup_scores(self):
        for p in self.popup_scores.parameters():
            p.requires_grad_(False)

    def unfreeze_popup_scores(self):
        for p in self.popup_scores.parameters():
            p.requires_grad_(True)

    def freeze_module_params(self):
        for p in self.module.parameters():
            p.requires_grad_(False)

    def unfreeze_module_params(self):
        for p in self.module.parameters():
            p.requires_grad_(True)

    def set_mask_training_mode(self):
        self.unfreeze_popup_scores()
        self.freeze_module_params()

    def set_subnetwork_training_mode(self):
        self.freeze_popup_scores()
        self.unfreeze_module_params()
    # -----------------------------------------------

    def _masked_parameters(self):
        masked_params = {}

        for name, param in self.module.named_parameters():
            # Only mask weight if just_weight=True
            if self.just_weight and name != "weight":
                masked_params[name] = param
                continue

            score = self.popup_scores[self._score_key(name)]
            mask = GetSubnet.apply(score.abs(), self.k)[:param.shape[0]]
            masked_params[name] = param * mask

        return masked_params

    def forward(self, x):
        masked_params = self._masked_parameters()
        buffers = dict(self.module.named_buffers())
        return functional_call(self.module, {**masked_params, **buffers}, (x,))


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