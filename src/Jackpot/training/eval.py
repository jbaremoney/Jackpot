import torch
from torch import nn, device


def test(split,
         model,
         train_loader_at_eval,
         test_loader,
         n_classes,
         return_metrics=False):

    model.eval()
    criterion = nn.CrossEntropyLoss()

    data_loader = train_loader_at_eval if split == 'train' else test_loader

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).long()   # shape [B]

            outputs = model(inputs)[:, :n_classes]   # shape [B, n_classes]
            loss = criterion(outputs, targets)

            preds = outputs.argmax(dim=1)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == targets).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples

    print(f"{split} loss: {avg_loss:.4f} acc: {acc:.4f}")

    if return_metrics:
        return avg_loss, acc

def evaluate_model(model, train_loader_at_eval, test_loader, task, n_classes, data_flag):
    train_loss, train_acc = test(
        split="train",
        model=model,
        train_loader_at_eval=train_loader_at_eval,
        test_loader=test_loader,
        n_classes=n_classes,
        return_metrics=True,
    )

    test_loss, test_acc = test(
        split="test",
        model=model,
        train_loader_at_eval=train_loader_at_eval,
        test_loader=test_loader,
        n_classes=n_classes,
        return_metrics=True,
    )

    return {
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
    }


def evaluate_at_epoch(model, epoch, train_loader_at_eval, test_loader, task, n_classes, data_flag):
    metrics = evaluate_model(
        model=model,
        train_loader_at_eval=train_loader_at_eval,
        test_loader=test_loader,
        task=task,
        n_classes=n_classes,
        data_flag=data_flag,
    )
    metrics["epoch"] = epoch
    return metrics