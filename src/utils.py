import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms


# 自定义转换：将10维数据相加变成28*28
class SumOverChannels(object):
    def __call__(self, tensor):
        return torch.sum(tensor, dim=0).unsqueeze(0)  # 对第一个维度求和，然后增加一个维度以保持2D形状


# 自定义转换：将第一个维度的10维数据打乱
class ShuffleChannels(object):
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, tensor):
        shuffled_order = self.rng.permutation(tensor.shape[0])  # 生成一个乱序的索引数组
        return tensor[shuffled_order]  # 使用乱序索引来重新排列通道


# 自定义转换：直接输出原始数据（实际上不需要任何操作）
class IdentityTransform(object):
    def __call__(self, tensor):
        return tensor

def compute_acc(pred, true):
    '''
        Compute the accuracy.
        @param pred: batch_size * num_classes
        @param true: batch_size
    '''
    return torch.mean((torch.argmax(pred, dim=1) == true).float()).item()
# dataset_sum = MyCustomDataset(transform=SumOverChannels())
# dataset_shuffle = MyCustomDataset(transform=ShuffleChannels(seed=42))  # 使用固定的随机种子以获得可重复的结果
# dataset_identity = MyCustomDataset(transform=IdentityTransform())

def get_val_acc(val_loader, model1, model2, DEVICE):
    model1.eval()  # 设置模型为评估模式
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

def base_train_process(NUM_EPOCHS, train_loader, DEVICE, optimizer,model2, model1, criterion, SAVE_EVERY, val_loader, name):
    best = 0
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
        # if (epoch + 1) % SAVE_EVERY == 0 and epoch > 50:  # 每隔SAVE_EVERY个epoch保存一次模型
        #     cur_acc = get_val_acc(val_loader, model1, model2, DEVICE)
        #     if cur_acc > best:
        #         model1_path = os.path.join('./models2/', name+'_model1_best.pth')
        #         torch.save(model1.state_dict(), model1_path)
        #         model2_path = os.path.join('./models2/', name+'_model2_best.pth')
        #         torch.save(model2.state_dict(), model2_path)
    return losses, acc, model1, model2

def free_data(*args):
    for i in args:
        i = None

def save(model1, model2, name, val_loader, DEVICE):
    torch.save(model1.state_dict(), './models/' + name + '_final_model1.pth')
    torch.save(model2.state_dict(), './models/' + name + '_final_model2.pth')
    final_acc = get_val_acc(val_loader, model1, model2, DEVICE)
    print(name + "final acc:{}".format(final_acc))
    free_data(model1, model2)

def compute_l2(XS, XQ):
    '''
        Compute the pairwise l2 distance
        @param XS (support x): support_size x ebd_dim
        @param XQ (support x): query_size x ebd_dim

        @return dist: query_size x support_size

    '''
    diff = XS.unsqueeze(0) - XQ.unsqueeze(1)
    dist = torch.norm(diff, dim=2)

    return dist ** 2

def draw_pic(*args, name):
    length = len(args)//2
    losses = args[:length]
    names = args[length:]
    # 创建一个新的figure对象
    plt.figure()
    # 绘制模型1的loss曲线，label参数用于设置图例名称
    for loss, line_name in zip(losses, names):
        plt.plot(loss, label=line_name)
    # 添加图例
    plt.legend()
    # 设置x轴标签
    plt.xlabel('Epochs')
    # 设置y轴标签
    if "ACC" in name:
        plt.ylabel("ACCURACY")
    else:
        plt.ylabel("LOSS")
    plt.savefig('./pic/{}.png'.format(name))
    plt.cla()

def draw_all(losses, accs):
    loss1, loss2, loss3, losses1, losses2, losses3, losses4 = losses
    acc1, acc2, acc3, acces1, acces2, acces3, acces4 = accs
    draw_pic(loss1, loss2, loss3, "BASE MODEL1", "BASE MODEL2", "BASE FAKE MODEL", name="BASE MODELS LOSS")
    draw_pic(acc1, acc2, acc3, "BASE MODEL1", "BASE MODEL2", "BASE FAKE MODEL", name="BASE MODELS ACCURACY")

    draw_pic(loss3, losses1, losses2, "BASE FAKE MODEL", "BASE DRO MODEL1", "BASE DRO MODEL2", name="BASE DRO MODELS' LOSS")
    draw_pic(acc3, acces1, acces2, "BASE FAKE MODEL", "BASE DRO MODEL1", "BASE DRO MODEL2", name="BASE DRO MODELS' ACCURACY")
    draw_pic(loss1, loss2, loss3, losses1, losses2, "BASE MODEL1", "BASE MODEL2", "BASE FAKE MODEL", "BASE DRO MODEL1", "BASE DRO MODEL2",
             name="COMPARE MODELS' LOSS")
    draw_pic(acc1, acc2, acc3, acces1, acces2, "BASE MODEL1", "BASE MODEL2", "BASE FAKE MODEL", "BASE DRO MODEL1", "BASE DRO MODEL2",
             name="COMPARE MODELS' ACCURACY")

    draw_pic(loss3, losses3, losses4, "BASE FAKE MODEL", "BASE DRO MODEL1", "BASE DRO MODEL2", name="BASE DRO MODELS' LOSS")
    draw_pic(acc3, acces3, acces4, "BASE FAKE MODEL", "BASE DRO MODEL1", "BASE DRO MODEL2", name="BASE DRO MODELS' ACCURACY")
    draw_pic(loss1, loss2, loss3, losses1, losses2, losses3, losses4, "BASE MODEL1", "BASE MODEL2", "BASE FAKE MODEL", "BASE DRO MODEL1",
             "BASE DRO MODEL2", "FINAL DRO MODEL1", "FINAL DRO MODEL2",
             name="COMPARE ALL MODELS' LOSS")
    draw_pic(acc1, acc2, acc3, acces1, acces2, acces3, acces4, "BASE MODEL1", "BASE MODEL2",  "BASE FAKE MODEL", "BASE DRO MODEL1", "BASE DRO MODEL2",
             "FINAL DRO MODEL1", "FINAL DRO MODEL2",
             name="COMPARE ALL MODELS' ACCURACY")

def draw_odd(losses, accs):
    loss1, loss2, loss3 = losses
    acc1, acc2, acc3 = accs
    draw_pic(loss1, loss2, loss3, "FAKE MODEL ON ODD", "CORRECT MODEL ON EVEN", "FINAL MODEL", name="MODELS LOSS")
    draw_pic(acc1, acc2, acc3, "FAKE MODEL ON ODD", "CORRECT MODEL ON EVEN", "FINAL MODEL", name="MODELS ACCURACY")