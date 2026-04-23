"""
Example: popupify a VGG-16 for CIFAR-10 and plot training loss and effective sparsity over training steps.

Run from repo root: ``PYTHONPATH=src python scripts/run_dbl_popup.py``
"""
from src.Jackpot.models.cifar import CIFARVGG16
from src.Jackpot.pruning.popup import popupify
from src.Jackpot.utils.utils import set_seed
from src.Jackpot.training.train import getTrainingDataLoaders, trainit
import torch.optim as optim
import matplotlib.pyplot as plt

DS_NAME = "cifar10"
NUM_EPOCHS = 60

set_seed()
model = CIFARVGG16()

set_seed()

# k = .5, training popup scores by default
pu_model = popupify(model)
optimizer = optim.Adam(filter(lambda param: param.requires_grad, pu_model.parameters()))

info, task, n_classes, train_loader, train_loader_at_eval, test_loader = getTrainingDataLoaders(
        DS_NAME,
        download=True,
        BATCH_SIZE=64,
        augment=True,
    )


prune_losses, prune_sparsities = trainit(
    pu_model,
    NUM_EPOCHS,
    train_loader,
    optimizer,
    task,
    n_classes,
    return_losses=True,
    return_sparsity=True,
)

plt.plot(prune_losses, label='Training loss')
plt.legend()
plt.title(f"Doubled popup training loss on cifar10")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
ax = plt.gca()
ax.set_ylim(top=1.0)        # only cap the top at 1.0
ax.get_xaxis().set_visible(False)


plt.figure(figsize=(8, 5))
plt.plot(prune_sparsities, label='Weight sparsity')
plt.legend()
plt.title(f"Effective weight sparsity over training steps on cifar10")
plt.xlabel("Training Steps")
plt.ylabel("Sparsity")
plt.show()
