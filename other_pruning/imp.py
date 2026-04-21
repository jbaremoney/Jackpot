from itertools import cycle
from torch import nn, optim
from models.layer import MaskLayer
import torch
import tqdm
import copy
from models.network import MaskedNetwork
from utils.training import trainit

# need new training function to give control over number of steps
def train_step(model, batch, optimizer, task, n_classes, device):
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    model.train()

    inputs, targets = batch
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)

    optimizer.zero_grad()
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

def train_for_steps(model,
                    num_steps,
                    train_loader,
                    optimizer,
                    task,
                    n_classes,
                    device,
                    return_losses=False,
                    no_progress=False):

    if return_losses:
        losses = []

    loader = cycle(train_loader)

    iterator = range(num_steps)
    if not no_progress:
        iterator = tqdm(iterator)

    for _ in iterator:
        batch = next(loader)
        loss_val = train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            task=task,
            n_classes=n_classes,
            device=device
        )

        if return_losses:
            losses.append(loss_val)

    if return_losses:
        return losses

# helper to get weight sparsity of MaskedNetwork
def maskednetwork_sparsity(masked_net):
    zero_count = 0
    total_count = 0

    for m in masked_net.modules():
        if isinstance(m, MaskLayer):
            mask = m.mask
            zero_count += (mask == 0).sum().item()
            total_count += mask.numel()

    if total_count == 0:
        raise ValueError("No MaskLayer modules found in masked_net")

    sparsity = zero_count / total_count
    return {
        "sparsity": sparsity,
        "zero_count": zero_count,
        "total_count": total_count,
    }



def get_mask_layers(model):
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
):
    """
    Iterative Magnitude Pruning:
    train -> prune -> rewind -> repeat
    """

    keep_ratio_final = 1.0 - final_sparsity
    if not (0.0 < keep_ratio_final <= 1.0):
        raise ValueError("final_sparsity must be in [0,1).")

    if L_max < 1:
        raise ValueError("L_max must be >= 1.")

    old_net = net
    base_net = copy.deepcopy(net).to(device)

    # Save initial weights before any training
    base_prunable_layers = [
        m for m in base_net.modules() if isinstance(m, (nn.Linear, nn.Conv2d))
    ]
    if len(base_prunable_layers) == 0:
        raise ValueError("No prunable nn.Linear or nn.Conv2d layers found.")

    init_weights = [layer.weight.detach().clone() for layer in base_prunable_layers]

    # Phase 1: optional rewind point training on unmasked network
    optimizer = optim.Adam(base_net.parameters())

    train_for_steps(
        base_net,
        tau,
        train_dataloader,
        optimizer,
        task,
        n_classes,
        device,
        return_losses=False,
        no_progress=False,
    )

    rewind_weights = [layer.weight.detach().clone() for layer in base_prunable_layers]

    # Wrap AFTER rewind point is obtained
    net = MaskedNetwork(base_net).to(device)
    mask_layers = get_mask_layers(net)

    # Ratio kept per round so that after L_max rounds you hit final sparsity
    round_keep_ratio = keep_ratio_final ** (1.0 / L_max)

    for level in range(L_max):
        print(f"IMP round {level + 1}/{L_max}")

        optimizer = optim.Adam(net.parameters())

        trainit(
            net,
            iter_epochs,
            train_dataloader,
            optimizer,
            task,
            n_classes,
            return_losses=False,
            no_progress=False,
        )

        if prune_global:
            # Global pruning over all currently active weights
            all_masked_mags = []
            all_refs = []

            for layer in mask_layers:
                mask = layer.mask
                weight = layer.weight.detach()
                masked_mags = weight.abs() * mask

                active_idx = (mask > 0).view(-1)
                active_vals = masked_mags.view(-1)[active_idx]

                all_masked_mags.append(active_vals)
                all_refs.append((layer, active_idx))

            all_scores = torch.cat(all_masked_mags)

            total_on = all_scores.numel()
            to_keep_total = int(round_keep_ratio * total_on)
            to_keep_total = max(to_keep_total, 0)

            if to_keep_total == 0:
                threshold = torch.inf
            else:
                sorted_vals, _ = torch.sort(all_scores)
                threshold = sorted_vals[-to_keep_total]

            for layer in mask_layers:
                mask = layer.mask
                masked_mags = layer.weight.detach().abs() * mask
                new_mask = ((masked_mags >= threshold) & (mask > 0)).to(mask.dtype)
                layer.mask.copy_(new_mask)

        else:
            # Layerwise pruning
            for layer in mask_layers:
                mask = layer.mask
                weight = layer.weight.detach()

                masked_mags = weight.abs() * mask
                flat_mask = mask.view(-1)
                flat_mags = masked_mags.view(-1)

                on_idx = torch.nonzero(flat_mask > 0, as_tuple=False).flatten()
                n_on = on_idx.numel()

                if n_on == 0:
                    continue

                to_keep = int(round_keep_ratio * n_on)
                to_keep = max(to_keep, 0)

                if to_keep == 0:
                    flat_mask[on_idx] = 0
                    continue

                active_mags = flat_mags[on_idx]
                _, order = torch.sort(active_mags)

                prune_idx = on_idx[order[:-to_keep]] if to_keep < n_on else torch.tensor([], device=on_idx.device, dtype=on_idx.dtype)
                flat_mask[prune_idx] = 0

        print(f"SPARSITY AFTER PRUNING LVL {level + 1}: {maskednetwork_sparsity(net)}")

        # Rewind surviving weights
        rewind_source = init_weights if rewind_to_init else rewind_weights
        for layer, rewind_w in zip(mask_layers, rewind_source):
            layer.weight.data.copy_(rewind_w.to(layer.weight.device) * layer.mask)

    # Build keep_masks keyed by original model modules
    keep_masks = {}
    old_modules = list(old_net.modules())

    raw_old_prunable = [m for m in old_modules if isinstance(m, (nn.Linear, nn.Conv2d))]

    for old_m, masked_layer in zip(raw_old_prunable, mask_layers):
        keep_masks[old_m] = masked_layer.mask.detach().clone()

    return keep_masks, net
