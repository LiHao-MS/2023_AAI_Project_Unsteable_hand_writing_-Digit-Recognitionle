import os

import numpy as np
import torch
from torch.utils.data import Dataset


class HandwrittenDigitsDataset(Dataset):
    """
       Custom PyTorch Dataset class for loading and preprocessing handwritten digit data.

       Args:
           root_dir (str): The directory path containing the dataset, where each subdirectory
                           represents a digit class.
           transform (callable, optional): A transformation function to be applied on the
                                           image tensors. Default is None.
           odd (bool, optional): If True, only includes odd digits (1, 3, 5, 7, 9) in the dataset.
                                 Default is False.
           even (bool, optional): If True, only includes even digits (0, 2, 4, 6, 8) in the dataset.
                                  Default is False.

       Note: Only one of `odd` or `even` can be set to True at a time. If both are False,
             all digits from 0 to 9 will be included.

       Attributes:
           root_dir (str): The base directory path of the dataset.
           transform (callable): The transformation function to apply to samples.
           classes (list): List of class names based on the 'odd' and 'even' parameters.
           data (list): A list of torch.Tensor instances representing the digit images.
           targets (list): A list of integer labels corresponding to the digit classes.

       Methods:
           __len__(): Returns the number of samples in the dataset.
           __getitem__(idx): Retrieves the i-th sample and its label from the dataset.
    """

    def __init__(self, root_dir, transform=None, odd=False, even=False):
        self.root_dir = root_dir
        self.transform = transform
        if not odd and not even:
            self.classes = [str(i) for i in range(10)]
        elif odd:
            self.classes = [str(i * 2 + 1) for i in range(5)]
        else:
            self.classes = [str(i * 2) for i in range(5)]
        self.data = []
        self.targets = []
        # Load and preprocess data from each class directory
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                numpy_array = np.load(file_path)
                tensor = torch.from_numpy(numpy_array)
                self.data.append(tensor)
                self.targets.append(int(class_name))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.targets[idx]
