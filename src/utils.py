import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms


# Custom data transformation class: Sums over the first dimension (channels) to create a 28x28 tensor
class SumOverChannels(object):
    def __call__(self, tensor):
        return torch.sum(tensor, dim=0).unsqueeze(0)


# Custom data transformation class: Randomly shuffles channels in the first dimension
class ShuffleChannels(object):
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, tensor):
        shuffled_order = self.rng.permutation(tensor.shape[0])  # 生成一个乱序的索引数组
        return tensor[shuffled_order]  # 使用乱序索引来重新排列通道


# Identity transform class that doesn't modify the input tensor
class IdentityTransform(object):
    def __call__(self, tensor):
        return tensor

def compute_acc(pred, true):
    '''
    Function to compute accuracy.

    Args:
        pred (torch.Tensor): Predictions of shape batch_size x num_classes.
        true (torch.Tensor): Ground truth labels of shape batch_size.

    Returns:
        float: Accuracy value.
    '''
    return torch.mean((torch.argmax(pred, dim=1) == true).float()).item()

def get_val_acc(val_loader, model1, model2, DEVICE):
    '''
        Function to evaluate models on validation set and calculate accuracy.

        Args:
            val_loader (torch DataLoader): Validation data loader.
            model1 (torch.nn.Module): First model in the pipeline.
            model2 (torch.nn.Module): Second model in the pipeline.
            DEVICE (torch.device): Device to run the models on.

        Returns:
            float: Validation accuracy.
    '''

    model1.eval()
    model2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):  # 迭代加载数据
            data, target = data.to(DEVICE), target.to(DEVICE)  # 将数据转移到正确的设备上
            outputs = model2(model1(data))
            correct += torch.sum((torch.argmax(outputs, dim=1) == target).float()).item()
            total += target.size(0)  # 累加总样本数
    return correct / total

def base_train_process(NUM_EPOCHS, train_loader, DEVICE, optimizer,model2, model1, criterion):
    '''
       Function to perform the basic training process for multiple epochs.

       Args:
           NUM_EPOCHS (int): Number of epochs to train for.
           train_loader (torch DataLoader): Training data loader.
           DEVICE (torch.device): Device to run the models on.
           optimizer (torch.optim.Optimizer): Optimizer for both models.
           model1 (torch.nn.Module): First model in the pipeline.
           model2 (torch.nn.Module): Second model in the pipeline.
           criterion (torch.nn.LossFunction): Loss function to optimize.

       Returns:
           tuple: Tuple containing lists of average losses and accuracies per epoch,
                  and the final trained models.
    '''

    losses = []  # 记录每个epoch的loss
    acc = []
    for epoch in range(NUM_EPOCHS):  # 训练循环
        model1.train()
        model2.train()
        train_loss = 0.0
        train_acc = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):  # 迭代加载数据
            data, target = data.to(DEVICE), target.to(DEVICE)  # 将数据转移到正确的设备上
            optimizer.zero_grad()  # 清空梯度缓存
            output = model2(model1(data))  # 前向传播
            loss = criterion(output, target)  # 计算损失
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新权重
            train_loss += loss.item()
            train_acc += torch.sum((torch.argmax(output, dim=1) == target).float()).item()
        losses.append(train_loss / len(train_loader))  # 计算并保存每个epoch的平均损失
        acc.append(train_acc / len(train_loader))
    return losses, acc, model1, model2

def free_data(*args):
    '''
        Function to free memory by setting variables to None.

        Args:
            *args: Any number of objects to be cleared from memory.
    '''

    for i in args:
        i = None

def save(model1, model2, name, val_loader, DEVICE):
    '''
       Function to save model states and print final validation accuracy.

       Args:
           models (tuple): Tuple containing two models (model1, model2).
           name (str): Name prefix for saving the models' state dictionaries.
           val_loader (torch DataLoader): Validation data loader.
           DEVICE (torch.device): Device to run the models on.
    '''

    torch.save(model1.state_dict(), './models/' + name + '_final_model1.pth')
    torch.save(model2.state_dict(), './models/' + name + '_final_model2.pth')
    final_acc = get_val_acc(val_loader, model1, model2, DEVICE)
    print(name + "final acc:{}".format(final_acc))
    free_data(model1, model2)

def compute_l2(XS, XQ):
    '''
        Function to compute pairwise L2 distance between sets of vectors.

        Args:
            XS (torch.Tensor): Support set embeddings of shape support_size x embedding_dim.
            XQ (torch.Tensor): Query set embeddings of shape query_size x embedding_dim.

        Returns:
            torch.Tensor: Pairwise distances matrix of shape query_size x support_size.
    '''
    diff = XS.unsqueeze(0) - XQ.unsqueeze(1)
    dist = torch.norm(diff, dim=2)

    return dist ** 2

def draw_pic(*args, name):
    '''
       Function to plot and save figures comparing multiple loss/accuracy curves.

       Args:
           *args: Alternating sequence of loss arrays and curve names.
           name (str): Base name for the saved figure file.
    '''
    length = len(args)//2
    losses = args[:length]
    names = args[length:]
    # create a new figure
    plt.figure()
    # plot the loss curves
    for loss, line_name in zip(losses, names):
        plt.plot(loss, label=line_name)
    plt.legend()
    plt.xlabel('Epochs')
    if "ACC" in name:
        plt.ylabel("ACCURACY")
    else:
        plt.ylabel("LOSS")
    plt.savefig('./pic/{}.png'.format(name))
    plt.cla()


def draw_all(losses, accs):
    '''
       Function to plot and save figures comparing multiple loss/accuracy curves.

       Args:
           losses (tuple): Tuple containing loss arrays for each model.
           accs (tuple): Tuple containing accuracy arrays for each model.
    '''
    loss1, loss2, loss3, losses1, losses2, losses3, losses4 = losses
    acc1, acc2, acc3, acces1, acces2, acces3, acces4 = accs
    draw_pic(loss1, loss2, loss3, "BASE MODEL1", "BASE MODEL2", "FAKE MODEL", name="BASE_MODELS_LOSS")
    draw_pic(acc1, acc2, acc3, "BASE MODEL1", "BASE MODEL2", "FAKE MODEL", name="BASE_MODELS_ACCURACY")

    draw_pic(loss3, losses1, losses2, "FAKE MODEL", "DRO MODEL1", "DRO MODEL2",
             name="BASE_DRO_MODELS_LOSS")
    draw_pic(acc3, acces1, acces2, "FAKE MODEL", "DRO MODEL1", "DRO MODEL2",
             name="BASE_DRO_MODELS_ACCURACY")
    draw_pic(loss1, loss2, loss3, losses1, losses2, "BASE MODEL1", "BASE MODEL2", "FAKE MODEL", "DRO MODEL1", "DRO MODEL2",
             name="COMPARE_MODELS_LOSS")
    draw_pic(acc1, acc2, acc3, acces1, acces2, "BASE MODEL1", "BASE MODEL2", "FAKE MODEL", "DRO MODEL1", "DRO MODEL2",
             name="COMPARE_MODELS_ACCURACY")

    draw_pic(loss3, losses3, losses4, "FAKE MODEL", "DRO MODEL1", "DRO MODEL2", name="DRO MODELS' LOSS")
    draw_pic(acc3, acces3, acces4, "FAKE MODEL", "DRO MODEL1", "DRO MODEL2", name="DRO MODELS' ACCURACY")
    draw_pic(loss1, loss2, loss3, losses1, losses2, losses3, losses4, "BASE MODEL1", "BASE MODEL2", "FAKE MODEL", "DRO MODEL1",
             "BASE DRO MODEL2", "TOFU MODEL1", "TOFU MODEL2",
             name="COMPARE_ALL_MODELS_LOSS")
    draw_pic(acc1, acc2, acc3, acces1, acces2, acces3, acces4, "BASE MODEL1", "BASE MODEL2",  "FAKE MODEL", "DRO MODEL1", "DRO MODEL2",
             "TOFU MODEL1", "TOFU MODEL2",
             name="COMPARE_ALL_MODELS_ACCURACY")


