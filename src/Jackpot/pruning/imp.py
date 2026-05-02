import copy
from itertools import cycle

import torch
from torch import nn, optim
from tqdm import tqdm

from src.Jackpot.models.masking import MaskLayer, MaskedNetwork
from src.Jackpot.training.train import trainit


def train_step(model, batch, optimizer, task, n_classes, device):
    """Run one optimizer step and return the scalar loss."""
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    model.train()

    inputs, targets = batch
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    optimizer.zero_grad()

    # Some shared architectures may output more logits than the current task needs.
    outputs = model(inputs)[:, :n_classes]

    if task == "multi-label, binary-class":
        targets = targets.to(torch.float32)
        loss = criterion(outputs, targets)
    else:
        targets = targets.long().view(-1)
        loss = criterion(outputs, targets)

    loss.backward()
    optimizer.step()

    return loss.item()


def train_for_steps(
    model,
    num_steps,
    train_loader,
    optimizer,
    task,
    n_classes,
    device,
    return_losses=False,
    no_progress=False,
):
    """Run a fixed number of optimizer steps by cycling through train_loader."""
    if num_steps < 0:
        raise ValueError(f"num_steps must be nonnegative, got {num_steps}")

    losses = [] if return_losses else None
    loader = cycle(train_loader)

    iterator = range(num_steps)
    if not no_progress:
        iterator = tqdm(iterator, desc="Training steps")

    for _ in iterator:
        loss_val = train_step(
            model=model,
            batch=next(loader),
            optimizer=optimizer,
            task=task,
            n_classes=n_classes,
            device=device,
        )

        if return_losses:
            losses.append(loss_val)

    if return_losses:
        return losses

    return None


def masked_network_sparsity(masked_net):
    """Return mask sparsity statistics across all MaskLayer modules."""
    zero_count = 0
    total_count = 0

    for m in masked_net.modules():
        if isinstance(m, MaskLayer):
            mask = m.mask
            zero_count += (mask == 0).sum().item()
            total_count += mask.numel()

    if total_count == 0:
        raise ValueError("No MaskLayer modules found in masked_net.")

    return {
        "sparsity": zero_count / total_count,
        "zero_count": zero_count,
        "total_count": total_count,
    }


def get_mask_layers(model):
    """Return all MaskLayer modules in traversal order."""
    return [m for m in model.modules() if isinstance(m, MaskLayer)]


def IMP(
    net,
    final_sparsity,
    train_dataloader,
    device,
    tau,
    L_max,
    iter_epochs,
    task,
    n_classes=10,
    train_fn=None,
    rewind_to_init=False,
    prune_global=False,
    verbose=False,
):
    """
    Iterative Magnitude Pruning.

    Repeats:
        train -> prune lowest-magnitude active weights -> rewind surviving weights.

    Args:
        net: Model to prune.
        final_sparsity: Target final fraction of pruned Conv2d/Linear weights.
        train_dataloader: Dataloader used for training during IMP.
        device: Device used for training/pruning.
        tau: Number of warmup steps used to obtain the rewind point.
        L_max: Number of pruning rounds.
        iter_epochs: Number of training epochs per pruning round.
        task: Task string used by the training function.
        n_classes: Number of output classes/logits to use.
        train_fn: Optional training function. Defaults to trainit.
        rewind_to_init: If True, rewind to initialization instead of tau-step weights.
        prune_global: If True, prune globally; otherwise prune layerwise.
        verbose: If True, print progress and sparsity statistics.

    Returns:
        (keep_masks, masked_net), where keep_masks maps original prunable modules
        to binary keep masks. A mask value of 1 means keep; 0 means prune.
    """
    if not 0.0 <= final_sparsity < 1.0:
        raise ValueError(f"final_sparsity must be in [0, 1), got {final_sparsity}")

    if L_max < 1:
        raise ValueError(f"L_max must be >= 1, got {L_max}")

    if tau < 0:
        raise ValueError(f"tau must be nonnegative, got {tau}")

    if iter_epochs < 0:
        raise ValueError(f"iter_epochs must be nonnegative, got {iter_epochs}")

    if train_fn is None:
        train_fn = trainit

    keep_ratio_final = 1.0 - final_sparsity
    old_net = net
    base_net = copy.deepcopy(net).to(device)

    base_prunable_layers = [
        m for m in base_net.modules()
        if isinstance(m, (nn.Linear, nn.Conv2d))
    ]

    if len(base_prunable_layers) == 0:
        raise ValueError("No prunable nn.Linear or nn.Conv2d layers found.")

    init_weights = [
        layer.weight.detach().clone()
        for layer in base_prunable_layers
    ]

    # Train to the rewind point.
    optimizer = optim.Adam(base_net.parameters())

    train_for_steps(
        model=base_net,
        num_steps=tau,
        train_loader=train_dataloader,
        optimizer=optimizer,
        task=task,
        n_classes=n_classes,
        device=device,
        return_losses=False,
        no_progress=not verbose,
    )

    rewind_weights = [
        layer.weight.detach().clone()
        for layer in base_prunable_layers
    ]

    # Wrap after the rewind point is obtained.
    net = MaskedNetwork(base_net).to(device)
    mask_layers = get_mask_layers(net)

    if len(mask_layers) != len(base_prunable_layers):
        raise RuntimeError(
            f"Expected {len(base_prunable_layers)} MaskLayer modules, "
            f"found {len(mask_layers)}."
        )

    # Per-round keep ratio chosen so repeated pruning reaches final_sparsity.
    round_keep_ratio = keep_ratio_final ** (1.0 / L_max)

    for level in range(L_max):
        if verbose:
            print(f"IMP round {level + 1}/{L_max}")

        optimizer = optim.Adam(net.parameters())

        train_fn(
            net,
            iter_epochs,
            train_dataloader,
            optimizer,
            task,
            n_classes,
            return_losses=False,
            no_progress=not verbose,
        )

        if prune_global:
            all_active_mags = []

            for layer in mask_layers:
                mask = layer.mask
                weight = layer.weight.detach()
                active_mags = weight.abs().reshape(-1)[mask.reshape(-1) > 0]
                all_active_mags.append(active_mags)

            all_scores = torch.cat(all_active_mags)
            total_on = all_scores.numel()

            if total_on == 0:
                raise RuntimeError("No active weights left to prune.")

            to_keep_total = max(int(round_keep_ratio * total_on), 1)

            if to_keep_total < total_on:
                sorted_vals, _ = torch.sort(all_scores)
                threshold = sorted_vals[-to_keep_total]

                for layer in mask_layers:
                    mask = layer.mask
                    masked_mags = layer.weight.detach().abs() * mask
                    new_mask = ((masked_mags >= threshold) & (mask > 0)).to(mask.dtype)
                    layer.mask.copy_(new_mask)

        else:
            for layer in mask_layers:
                mask = layer.mask
                weight = layer.weight.detach()

                masked_mags = weight.abs() * mask
                flat_mask = mask.reshape(-1)
                flat_mags = masked_mags.reshape(-1)

                on_idx = torch.nonzero(flat_mask > 0, as_tuple=False).flatten()
                n_on = on_idx.numel()

                if n_on == 0:
                    continue

                to_keep = max(int(round_keep_ratio * n_on), 1)

                if to_keep < n_on:
                    active_mags = flat_mags[on_idx]
                    _, order = torch.sort(active_mags)

                    prune_idx = on_idx[order[:-to_keep]]
                    flat_mask[prune_idx] = 0

        if verbose:
            stats = masked_network_sparsity(net)
            print(f"Sparsity after pruning round {level + 1}: {stats}")

        # Rewind surviving weights.
        rewind_source = init_weights if rewind_to_init else rewind_weights

        with torch.no_grad():
            for layer, rewind_w in zip(mask_layers, rewind_source):
                layer.weight.copy_(rewind_w.to(layer.weight.device) * layer.mask)

    keep_masks = {}

    old_prunable_layers = [
        m for m in old_net.modules()
        if isinstance(m, (nn.Linear, nn.Conv2d))
    ]

    if len(old_prunable_layers) != len(mask_layers):
        raise RuntimeError(
            f"Expected {len(old_prunable_layers)} original prunable layers, "
            f"found {len(mask_layers)} mask layers."
        )

    for old_layer, masked_layer in zip(old_prunable_layers, mask_layers):
        keep_masks[old_layer] = masked_layer.mask.detach().clone()

    return keep_masks, net