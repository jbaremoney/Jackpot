import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import autograd
from torch.func import functional_call


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


class MaskLayer(nn.Module):
    """used for comparison to snip, grasp, imp
        only masks weights
    """
    def __init__(self, layer: nn.Module, mask: torch.Tensor | None = None):
        super().__init__()

        if not isinstance(layer, (nn.Linear, nn.Conv2d)):
            raise TypeError("MaskLayer only supports nn.Linear and nn.Conv2d")

        self.layer = layer

        if mask is None:
            mask = torch.ones_like(layer.weight)
        else:
            mask = mask.detach().clone().to(layer.weight.device, dtype=layer.weight.dtype)

        if mask.shape != layer.weight.shape:
            raise ValueError(
                f"Mask shape {mask.shape} must match weight shape {layer.weight.shape}"
            )

        self.register_buffer("mask", mask)

    @property
    def weight(self):
        return self.layer.weight

    @property
    def bias(self):
        return self.layer.bias

    def forward(self, x):
        masked_weight = self.layer.weight * self.mask
        bias = self.layer.bias

        if isinstance(self.layer, nn.Linear):
            return F.linear(x, masked_weight, bias)

        elif isinstance(self.layer, nn.Conv2d):
            return F.conv2d(
                x,
                masked_weight,
                bias,
                stride=self.layer.stride,
                padding=self.layer.padding,
                dilation=self.layer.dilation,
                groups=self.layer.groups,
            )

        raise RuntimeError("Unsupported layer type in MaskLayer")