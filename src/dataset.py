import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os


class HandwrittenDigitsDataset(Dataset):
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
        # 遍历每个目录，加载数据
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                numpy_array = np.load(file_path)  # 加载numpy数组
                tensor = torch.from_numpy(numpy_array)  # 将numpy数组转换为torch张量
                self.data.append(tensor)
                self.targets.append(int(class_name))  # 将类名转换为整数标签

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, self.targets[idx]
