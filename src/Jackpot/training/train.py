import torch
from torch import nn, device
import tqdm
from src.Jackpot.training.eval import evaluate_at_epoch
from src.Jackpot.utils.utils import get_effective_sparsity_info


def trainit(model,
            NUM_EPOCHS,
            train_loader,
            optimizer,
            task,
            n_classes,
            return_losses=False,
            no_progress=False,
            return_sparsity=False):

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if return_losses:
        losses = []

    if return_sparsity:
        sparsities = []

    for epoch in range(NUM_EPOCHS):
        print(f"{epoch / NUM_EPOCHS * 100:.1f}% DONE")
        model.train()

        loader = train_loader if no_progress else tqdm(train_loader)

        for inputs, targets in loader:
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

            if return_losses:
                losses.append(loss.item())

            if return_sparsity:
                sparsities.append(get_effective_sparsity_info(model)["sparsity"])

    if return_losses and return_sparsity:
        return losses, sparsities
    elif return_losses:
        return losses
    elif return_sparsity:
        return sparsities



def train_with_epoch_checkpoints(
    model,
    total_epochs,
    checkpoint_epochs_input, # Renamed parameter for clarity
    train_loader,
    optimizer,
    task,
    n_classes,
    train_loader_at_eval,
    test_loader,
    data_flag,
    run_dir,
    prefix="train",
    no_progress=False,
    return_losses=True,
    return_sparsity=True,
):
    """
    Train in chunks so we can evaluate at specific epoch counts.

    This is designed to match the behavior of repeated calls to your
    original trainit(...), while keeping the same optimizer/model state.

    Returns:
        all_losses, all_sparsities, checkpoint_rows
    """

    # Ensure unique and sorted checkpoints, including 0 if present
    checkpoint_epochs_input = sorted(list(set(checkpoint_epochs_input)))

    all_losses = [] if return_losses else None
    all_sparsities = [] if return_sparsity else None
    checkpoint_rows = []

    trained_so_far = 0

    # Handle epoch 0 evaluation separately if requested
    if 0 in checkpoint_epochs_input:
        print("Evaluating model at epoch 0 (pre-training)...")
        metrics = evaluate_at_epoch(
            model=model,
            epoch=0,
            train_loader_at_eval=train_loader_at_eval,
            test_loader=test_loader,
            task=task,
            n_classes=n_classes,
            data_flag=data_flag,
        )
        checkpoint_rows.append(metrics)


    # Filter for positive checkpoints for the training loop
    positive_checkpoints = sorted(set(e for e in checkpoint_epochs_input if e > 0 and e <= total_epochs))
    if not positive_checkpoints or positive_checkpoints[-1] != total_epochs:
        positive_checkpoints.append(total_epochs)
        positive_checkpoints = sorted(list(set(positive_checkpoints))) # Sort again to maintain order

    for target_epoch in positive_checkpoints:
        chunk_epochs = target_epoch - trained_so_far
        if chunk_epochs <= 0:
            continue

        # call trainit using the same flags/behavior as your original function
        out = trainit(
            model,
            chunk_epochs,
            train_loader,
            optimizer,
            task=task,
            n_classes=n_classes,
            return_losses=return_losses,
            no_progress=no_progress,
            return_sparsity=return_sparsity
        )

        # unpack exactly according to trainit's return style
        if return_losses and return_sparsity:
            chunk_losses, chunk_sparsities = out
            if all_losses is not None: all_losses.extend(chunk_losses)
            if all_sparsities is not None: all_sparsities.extend(chunk_sparsities)
        elif return_losses:
            chunk_losses = out
            if all_losses is not None: all_losses.extend(chunk_losses)
        elif return_sparsity:
            chunk_sparsities = out
            if all_sparsities is not None: all_sparsities.extend(chunk_sparsities)

        trained_so_far = target_epoch

        metrics = evaluate_at_epoch(
            model=model,
            epoch=trained_so_far,
            train_loader_at_eval=train_loader_at_eval,
            test_loader=test_loader,
            task=task,
            n_classes=n_classes,
            data_flag=data_flag
        )
        checkpoint_rows.append(metrics)

    return all_losses, all_sparsities, checkpoint_rows