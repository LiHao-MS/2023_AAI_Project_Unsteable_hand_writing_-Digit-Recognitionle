import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        '''
            @param x: batch_size * ebd_dim
            @param y: batch_size

            @return acc
            @return loss
        '''
        x = self.dropout(x)
        x = self.fc(x)
        x = F.tanh(x)
        # x = torch.log_softmax(x, dim=-1)

        # if y is None:
        #     # return prediction directly
        #     return torch.argmax(x, dim=1)

        # loss = F.cross_entropy(x, y)
        # acc = self.compute_acc(x, y)

        return x

    @staticmethod
    def compute_acc(pred, true):
        '''
            Compute the accuracy.
            @param pred: batch_size * num_classes
            @param true: batch_size
        '''
        return torch.mean((torch.argmax(pred, dim=1) == true).float()).item()
