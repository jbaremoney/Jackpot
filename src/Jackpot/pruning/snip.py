"""
SNIP: single-shot network pruning using |w * dL/dw| saliency.

Based on the reference implementation lineage:
https://github.com/JingtongSu/sanity-checking-pruning/blob/main/pruner/SNIP.py
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def SNIP_fetch_data(dataloader, num_classes, samples_per_class):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]

    found_classes = set()
    dataloader_iter = iter(dataloader)

    while True:
        try:
            inputs, targets = next(dataloader_iter)
        except StopIteration as exc:
            raise ValueError(
                "Could not collect samples_per_class examples for every class. "
                "Try lowering samples_per_class or check that the dataloader contains "
                "enough examples from each class."
            ) from exc

        for idx in range(inputs.shape[0]):
            x = inputs[idx : idx + 1]
            y = targets[idx : idx + 1]

            category = y.item()

            if category < 0 or category >= num_classes:
                raise ValueError(
                    f"Target label {category} is outside expected range "
                    f"[0, {num_classes - 1}]."
                )

            if len(datas[category]) == samples_per_class:
                found_classes.add(category)
                continue

            datas[category].append(x)
            labels[category].append(y)

            if len(datas[category]) == samples_per_class:
                found_classes.add(category)

        if len(found_classes) == num_classes:
            break

    X = torch.cat([torch.cat(class_data, dim=0) for class_data in datas], dim=0)
    y = torch.cat([torch.cat(class_labels, dim=0) for class_labels in labels], dim=0)
    y = y.view(-1)

    return X, y


def SNIP(
    net,
    ratio,
    train_dataloader,
    device,
    num_classes=10,
    samples_per_class=25,
    num_iters=1,
    verbose=False,
):
    """
    Compute SNIP keep masks for Conv2d and Linear weights.

    SNIP scores each weight by |w * dL/dw| on a small balanced batch.

    Args:
        net: Model to prune.
        ratio: Fraction of Conv2d/Linear weights to prune.
        train_dataloader: Dataloader used to sample scoring batches.
        device: Device used for scoring.
        num_classes: Number of classes.
        samples_per_class: Number of examples collected per class.
        num_iters: Number of balanced batches used to accumulate gradients.
        verbose: If True, print scoring information.

    Returns:
        Dictionary mapping modules in the original model to binary keep masks.
        A mask value of 1 means keep the weight; 0 means prune it.
    """
    if not 0 <= ratio <= 1:
        raise ValueError(f"ratio must be between 0 and 1, got {ratio}")

    if num_iters < 1:
        raise ValueError(f"num_iters must be >= 1, got {num_iters}")

    eps = 1e-10
    keep_ratio = 1.0 - ratio
    old_net = net

    # Score a copy so that the original model is not mutated.
    net = copy.deepcopy(net).to(device)
    net.zero_grad()

    weights = []

    for layer in net.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            layer.weight.requires_grad_(True)
            weights.append(layer.weight)

    if len(weights) == 0:
        raise ValueError("SNIP found no Conv2d or Linear weights to score.")

    # Accumulate gradients over one or more balanced batches.
    for it in range(num_iters):
        if verbose:
            print(f"(SNIP): Iteration {it + 1}/{num_iters}")

        inputs, targets = SNIP_fetch_data(
            train_dataloader,
            num_classes,
            samples_per_class,
        )

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = net(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()

    # Build SNIP saliency scores for the original model's modules.
    grads = {}
    old_modules = list(old_net.modules())

    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            if layer.weight.grad is None:
                raise RuntimeError("Expected weight gradients after SNIP backward pass.")

            old_layer = old_modules[idx]
            grads[old_layer] = torch.abs(layer.weight.detach() * layer.weight.grad)

    # Gather all scores into one vector and normalize.
    all_scores = torch.cat([score.flatten() for score in grads.values()])
    norm_factor = torch.abs(torch.sum(all_scores)) + eps
    all_scores = all_scores / norm_factor

    # Number of parameters to keep.
    num_params_to_keep = int(len(all_scores) * keep_ratio)

    if num_params_to_keep <= 0:
        return {
            module: torch.zeros_like(score)
            for module, score in grads.items()
        }

    if num_params_to_keep >= len(all_scores):
        return {
            module: torch.ones_like(score)
            for module, score in grads.items()
        }

    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    if verbose:
        print("** norm factor:", norm_factor)
        print("** accept:", acceptable_score)

    # Keep weights with scores at or above the global threshold.
    keep_masks = {}

    for module, score in grads.items():
        keep_masks[module] = ((score / norm_factor) >= acceptable_score).float()

    if verbose:
        num_kept = sum(mask.sum().item() for mask in keep_masks.values())
        print(f"** kept parameters: {num_kept}")

    return keep_masks