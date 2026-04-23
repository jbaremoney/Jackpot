"""Dataset utilities for faster epochs when transforms are fixed."""

from torch.utils.data import Dataset


class PreloadedDataset(Dataset):
    """Materialize every ``(image, label)`` once so each epoch avoids per-item transforms."""

    def __init__(self, dataset):
        self.images = []
        self.labels = []

        for i in range(len(dataset)):
            # DOES TRANSFORM UP FRONT BEFORE TRAINING
            img, label = dataset[i]

            self.images.append(img)
            self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]