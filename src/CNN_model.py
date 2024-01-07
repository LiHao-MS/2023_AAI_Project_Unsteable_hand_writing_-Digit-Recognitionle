import torch.nn as nn
import torch.nn.functional as F
import torch


class CnnM(nn.Module):
    """
      A simple Convolutional Neural Network (CNN) model.

      Args:
          input_channel (int): The number of input channels in the input image tensor.
          hidden_dim (int): The output dimensionality of the fully connected layer.

      Attributes:
          conv1 (nn.Conv2d): 2D convolutional layer with 32 output channels, 3x3 kernel size,
                             and stride of 1. No padding is applied.
          conv2 (nn.Conv2d): Another 2D convolutional layer with 64 output channels,
                             3x3 kernel size, and stride of 1. No padding is applied.
          conv3 (nn.Conv2d): Optional third 2D convolutional layer that is not used in forward pass.
          dropout1 (nn.Dropout): Dropout layer with a drop probability of 0.1 for regularization.
          fc1 (nn.Linear): Fully connected linear layer to project flattened features into a hidden space.
          # fc2 (nn.Linear): An unused fully connected linear layer which could be employed
                            for classification purposes with 10 output units.

      Methods:
          forward(x): Defines the forward pass through the network.
      """
    def __init__(self, input_channel, hidden_dim):
        super(CnnM, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 32, 3, 1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=0)

        self.conv3 = nn.Conv2d(64, 32, 3, 1, padding=0)
        self.dropout1 = nn.Dropout(0.1)

        self.fc1 = nn.Linear(64*12*12, hidden_dim)


    def forward(self, x):
        """
            Forward pass through the network.

            Args:
                x (torch.Tensor): Input data tensor of shape [batch_size, input_channel, height, width].

            Returns:
                torch.Tensor: Output tensor after passing through the network. Shape depends on `hidden_dim`.
        """
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
