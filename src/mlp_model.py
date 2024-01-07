import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
        A simple Multi-Layer Perceptron (MLP) model with dropout regularization.

        Args:
            input_dim (int): The number of input features.
            output_dim (int): The number of output classes.

        Attributes:
            dropout (nn.Dropout): Dropout layer for regularization to prevent overfitting.
            fc (nn.Linear): Fully connected linear layer for transforming input features into logits.
    """

    def __init__(self, input_dim, output_dim):
        # Initialize the parent nn.Module class
        super(MLP, self).__init__()

        # Define the layers and parameters of the model
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
            Forward propagation through the MLP.

            Args:
                x (torch.Tensor): Input data tensor with shape [batch_size, input_dim].

            Returns:
                x (torch.Tensor): Output tensor with log-softmax scores for each class
                with shape [batch_size, output_dim].
        """
        x = self.dropout(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
