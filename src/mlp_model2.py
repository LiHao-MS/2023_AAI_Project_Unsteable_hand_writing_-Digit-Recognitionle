import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x, y=None, return_pred=False, grad_penalty=False,
                return_logit=False):
        '''
            @param x: batch_size * ebd_dim
            @param y: batch_size

            @return acc
            @return loss
        '''
        x = self.dropout(x)
        x = self.fc(x)
        # x = F.tanh(x)
        x = F.softmax(x)

        return x

    @staticmethod
    def compute_acc(pred, true):
        '''
            Compute the accuracy.
            @param pred: batch_size * num_classes
            @param true: batch_size
        '''
        return torch.mean((torch.argmax(pred, dim=1) == true).float()).item()
