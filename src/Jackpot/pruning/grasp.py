"""
**GraSP**: gradient signal preservation via Hessian–gradient structure on a mini-batch.

Based on the reference `GraSP pruner
<https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py>`_.
"""
import copy

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


def GraSP_fetch_data(dataloader, num_classes, samples_per_class):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx+1], targets[idx:idx+1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
    return X, y


def GraSP(
    net,
    ratio,
    train_dataloader,
    device,
    num_classes=10,
    samples_per_class=25,
    num_iters=1,
    T=200,
    reinit=False,
):
    """
    Compute GraSP keep masks for Conv2d and Linear weights.

    GraSP scores each weight using a second-order gradient-preservation criterion.
    In this implementation, the score is approximately ``-w * (H g)``, where ``g``
    is an accumulated gradient vector and ``H g`` is a Hessian-gradient product
    estimated from balanced training batches.

    Args:
        net: Model to prune.
        ratio: Fraction of Conv2d/Linear weights to prune.
        train_dataloader: Dataloader used to sample scoring batches.
        device: Device used for scoring.
        num_classes: Number of classes.
        samples_per_class: Number of examples collected per class.
        num_iters: Number of balanced batches used to accumulate gradient signal.
        T: Temperature applied to logits before cross-entropy.
        reinit: If True, reinitialize scored Conv2d/Linear weights before scoring.

    Returns:
        Dictionary mapping modules in the original model to binary keep masks.
        A mask value of 1 means keep the weight; 0 means prune it.
    """
    eps = 1e-10
    old_net = net

    net = copy.deepcopy(net).to(device)
    net.zero_grad()

    weights = []

    for layer in net.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            if reinit:
                # Optionally reinitialize scored weights before computing GraSP scores.
                nn.init.xavier_normal_(layer.weight)
            weights.append(layer.weight)

    # Store data chunks for the second-order phase.
    stored_inputs = []
    stored_targets = []

    # will be list of gradient tensors
    grad_w = None
    for w in weights:
        w.requires_grad_(True)

    for it in range(num_iters):
        print("(1): Iteration %d/%d." % (it + 1, num_iters))

        inputs, targets = GraSP_fetch_data(
            train_dataloader,
            num_classes,
            samples_per_class
        )
        N = inputs.shape[0]

        first_inputs = inputs[:N // 2]
        first_targets = targets[:N // 2]

        second_inputs = inputs[N // 2:]
        second_targets = targets[N // 2:]

        # Store CPU copies/chunks for the second-order phase later.
        stored_inputs.append(first_inputs)
        stored_targets.append(first_targets)
        stored_inputs.append(second_inputs)
        stored_targets.append(second_targets)

        # First half gradient.
        first_inputs = first_inputs.to(device)
        first_targets = first_targets.to(device)

        outputs = net(first_inputs) / T
        loss = F.cross_entropy(outputs, first_targets)

        grad_w_p = autograd.grad(loss, weights)

        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

        # Second half gradient.
        second_inputs = second_inputs.to(device)
        second_targets = second_targets.to(device)

        outputs = net(second_inputs) / T
        loss = F.cross_entropy(outputs, second_targets)

        grad_w_p = autograd.grad(loss, weights, create_graph=False)

        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

    # Second-order GraSP phase.
    # This computes a Hessian-gradient-style signal in layer.weight.grad.
    net.zero_grad()

    num_stored_chunks = len(stored_inputs)

    for it in range(num_stored_chunks):
        print("(2): Iterations %d/%d." % (it + 1, num_stored_chunks))

        inputs = stored_inputs.pop(0).to(device)
        targets = stored_targets.pop(0).to(device)

        outputs = net(inputs) / T
        loss = F.cross_entropy(outputs, targets)

        # Differentiable gradient of this chunk's loss wrt each weight tensor.
        grad_f = autograd.grad(loss, weights, create_graph=True)

        # Inner product <grad_w, grad_f>.
        # Backpropagating through this gives a Hessian-gradient product.
        z = 0
        for grad_w_layer, grad_f_layer in zip(grad_w, grad_f):
            z += (grad_w_layer.detach() * grad_f_layer).sum()

        z.backward()


    # Build GraSP scores for the original model's modules.
    grads = {}
    old_modules = list(old_net.modules())

    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            old_layer = old_modules[idx]
            grads[old_layer] = -layer.weight.detach() * layer.weight.grad


    # Gather all scores into one vector and normalize.
    all_scores = torch.cat([score.flatten() for score in grads.values()])
    norm_factor = torch.abs(torch.sum(all_scores)) + eps
    all_scores = all_scores / norm_factor


    # Find global pruning threshold.
    num_params_to_prune = int(len(all_scores) * ratio)

    if num_params_to_prune == 0:
        return {
            module: torch.ones_like(score)
            for module, score in grads.items()
        }

    if num_params_to_prune >= len(all_scores):
        return {
            module: torch.zeros_like(score)
            for module, score in grads.items()
        }

    threshold, _ = torch.topk(all_scores, num_params_to_prune, sorted=True)
    acceptable_score = threshold[-1]

    # Build binary masks.
    keep_masks = {}

    for module, score in grads.items():
        keep_masks[module] = ((score / norm_factor) <= acceptable_score).float()

    return keep_masks