import torch.nn as nn
import torch.nn.functional as F
import torch


class HandwrittenDigitModel(nn.Module):
    def __init__(self):
        super(HandwrittenDigitModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=0)
        self.conv3 = nn.Conv2d(64, 32, 3, 1, padding=0)
        self.dropout1 = nn.Dropout(0.15)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(3200, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.sigmoid(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.sigmoid(x)
        print(x.shape)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output