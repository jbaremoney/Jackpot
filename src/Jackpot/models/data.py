from torch.utils.data import Dataset

# wrap dataset with this if we don't transform during training
# speeds up training a lot
class PreloadedDataset(Dataset):
    def __init__(self, dataset):
        self.images = []
        self.labels = []

        for i in range(len(dataset)):
            # DOES TRANSFORM UP FRONT BEFORE TRAINING
            # this is why the training loop goes faster
            img, label = dataset[i]

            self.images.append(img)
            self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]