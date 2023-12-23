import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os


class HandwrittenDigitsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = sorted(os.listdir(root_dir))
        self.data = []

        for label in self.labels:
            label_dir = os.path.join(root_dir, label)
            files = os.listdir(label_dir)
            for file in files:
                file_path = os.path.join(label_dir, file)
                npy_data = np.load(file_path)
                collapsed_data = np.sum(npy_data, axis=0)  # 10*28*28 -> 28*28
                collapsed_data = collapsed_data[np.newaxis, :, :]
                self.data.append((collapsed_data, int(label)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label