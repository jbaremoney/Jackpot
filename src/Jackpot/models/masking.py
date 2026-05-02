"""
Explicit binary masks on ``nn.Linear`` / ``nn.Conv2d`` weights.

Used to extract a fixed subnetwork (e.g. masks from SNIP, GraSP, or IMP) while
keeping the same forward pass as the original layer.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskLayer(nn.Module):
    """Wraps a linear or conv layer as ``y = f(x; w ⊙ m)`` with a stored binary mask ``m``."""

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


class MaskedNetwork(nn.Module):
    """Deep-copies ``net`` and replaces each ``Linear``/``Conv2d`` with a ``MaskLayer``."""

    def __init__(self, net: nn.Module, masks=None):
        super().__init__()

        self.net = copy.deepcopy(net)

        if masks is None:
            masks = []

        self._mask_index = 0
        self._provided_masks = masks

        self._wrap_modules(self.net)


    def _next_mask(self):
        if self._mask_index < len(self._provided_masks):
            mask = self._provided_masks[self._mask_index]
        else:
            mask = None
        self._mask_index += 1
        return mask

    def _wrap_modules(self, module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, (nn.Linear, nn.Conv2d)):
                mask = self._next_mask()
                setattr(module, name, MaskLayer(child, mask))
            else:
                self._wrap_modules(child)

    def forward(self, x):
        return self.net(x)

    def get_masks(self):
        """Return mask tensors in module traversal order (one per ``MaskLayer``)."""
        masks = []
        for m in self.net.modules():
            if isinstance(m, MaskLayer):
                masks.append(m.mask)
        return masks


