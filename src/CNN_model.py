import torch.nn as nn
import torch.nn.functional as F
import torch


class CnnM(nn.Module):
    def __init__(self, input_channel, hidden_dim):
        super(CnnM, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 32, 3, 1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=0)

        self.conv3 = nn.Conv2d(64, 32, 3, 1, padding=0)
        self.dropout1 = nn.Dropout(0.1)
        # self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64*12*12, hidden_dim)
        # self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        return x
