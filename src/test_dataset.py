import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os


class TestDataset(Dataset):
    """
        A custom dataset class to load and handle .npy files for training or testing.

        Args:
            root_dir (str): The directory path containing the .npy data files.
            transform (callable, optional): A function/transform that takes in a sample
                and applies transformations. Defaults to None.
            odd (bool, optional): If True, only include odd-indexed files. Default is False.
            even (bool, optional): If True, only include even-indexed files. Default is False.

        Attributes:
            root_dir (str): Directory path of the dataset.
            transform (callable): Data transformation function if provided.
            data (list): List of loaded and converted torch tensors from .npy files.
            target (list): List of file names corresponding to each tensor in 'data'.
    """
    def __init__(self, root_dir, transform=None, odd=False, even=False):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.target = []
        # Iterate through all files in the specified directory
        for file_name in os.listdir(root_dir):
            file_path = os.path.join(root_dir, file_name)
            # Check if the file is a .npy file
            if file_name.endswith(".npy"):
                try:
                    numpy_array = np.load(file_path)  # Load numpy array
                    tensor = torch.from_numpy(numpy_array)  # Convert numpy array to torch tensor
                    self.data.append(tensor)
                    self.target.append(file_name)
                except EOFError:
                    print(f"Warning: File {file_name} could not be loaded due to 'EOFError'. Skipping this file.")
                except Exception as e:
                    print(
                        f"Warning: An unexpected error occurred while loading file {file_name}: {e}. Skipping this file.")
            else:
                print(f"Warning: Skipped non-.npy file: {file_name}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.target[idx]
        return sample, target
