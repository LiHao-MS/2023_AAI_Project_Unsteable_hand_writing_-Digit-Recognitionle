import torch.nn as nn
import torchvision.transforms as transforms

# __all__ = ['MNIST', 'CIFAR10']


class MLP(nn.Module):
    def __init__(self, input=28 * 28, num_hidden=512, num_classes=10):
        super(MLP, self).__init__()
        self.input = input
        self.num_hidden = num_hidden
        self.layer0 = nn.Linear(self.input, self.num_hidden)
        self.layer1 = nn.Linear(self.num_hidden, self.num_hidden)
        self.layer2 = nn.Linear(self.num_hidden, self.num_hidden)
        self.layer3 = nn.Linear(self.num_hidden, num_classes)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        # x = x.view(-1, self.input)
        x = nn.functional.relu(self.layer0(x))
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = self.layer3(x)

        return nn.functional.log_softmax(x, dim=0)


# if dataset == 'mnist':
#     self.input = 28 * 28
# elif dataset == 'cifar10':
#     self.input = 3 * 32 * 32
#
# class MNIST:
#     base = MLP
#     args = list()
#     kwargs = {'input': 28 * 28, 'num_classes': 10}
#     transform_train = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])
#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])


# class CIFAR10:
#     base = MLP
#     args = list()
#     kwargs = {'input': 3 * 32 * 32, 'num_classes': 10}
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
#     ])
#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
#     ])
