from dataset import HandwrittenDigitsDataset
from CNN_model import CnnM
from mlp_model import MLP
import sys
import json
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from utils import *
from DRO import *

BATCH_SIZE = 100  # 根据需要调整batch大小
LEARNING_RATE = 0.001  # 学习率
NUM_EPOCHS = 1  # 根据需要调整epoch数量
WEIGHT_DECAY = 0.01  # 学习率衰减
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_EVERY = 10


def train1():
    name = "type1"
    model1 = CnnM(input_channel=1, hidden_dim=300).to(DEVICE)
    model2 = MLP(input_dim=300, output_dim=10).to(DEVICE)
    train_dataset = HandwrittenDigitsDataset('../processed_data/train', transform=SumOverChannels())
    val_dataset = HandwrittenDigitsDataset("../processed_data/val", transform=SumOverChannels())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    criterion = F.cross_entropy
    losses, acc, model1, model2 = base_train_process(NUM_EPOCHS, train_loader, DEVICE, optimizer, model2, model1,
                                                     criterion, SAVE_EVERY, val_loader, name)
    # 训练完成后，保存模型
    save(model1, model2, name, val_loader, DEVICE)
    free_data(train_dataset, train_loader, val_dataset, val_loader)
    return losses, acc


def train2():
    name = "type2"
    model1 = CnnM(input_channel=10, hidden_dim=300).to(DEVICE)
    model2 = MLP(input_dim=300, output_dim=10).to(DEVICE)
    train_dataset = HandwrittenDigitsDataset('../processed_data/train', transform=ShuffleChannels(seed=42))
    val_dataset = HandwrittenDigitsDataset("../processed_data/val", transform=ShuffleChannels(seed=42))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    criterion = F.cross_entropy
    losses, acc, model1, model2 = base_train_process(NUM_EPOCHS, train_loader, DEVICE, optimizer, model2, model1,
                                                     criterion,
                                                     SAVE_EVERY, val_loader, name)
    # 训练完成后，保存模型
    save(model1, model2, name, val_loader, DEVICE)
    free_data(train_dataset, train_loader, val_dataset, val_loader)
    return losses, acc


def train3():
    name = "type3"
    model1 = CnnM(input_channel=10, hidden_dim=300).to(DEVICE)
    model2 = MLP(input_dim=300, output_dim=10).to(DEVICE)
    model1.load_state_dict(torch.load('./models/type2_model1_best.pth', map_location=DEVICE))
    model2.load_state_dict(torch.load('./models/type2_model2_best.pth', map_location=DEVICE))
    models = [model1, model2]
    train_dataset = HandwrittenDigitsDataset('../processed_data/train', transform=ShuffleChannels(seed=42))
    val_dataset = HandwrittenDigitsDataset("../processed_data/val", transform=ShuffleChannels(seed=42))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    loaders = get_two_class_loaders(models, train_loader, DEVICE)
    losses1, acces1, model1, model2 = dro_train_process(loaders, models, optimizer, DEVICE, NUM_EPOCHS)
    model3 = CnnM(input_channel=10, hidden_dim=300).to(DEVICE)
    model4 = MLP(input_dim=300, output_dim=10).to(DEVICE)
    models = [model3, model4]
    losses2, acces2, model1, model2 = dro_train_process(loaders, models, optimizer, DEVICE, NUM_EPOCHS)
    save(model1, model2, name+"v1", val_loader, DEVICE)
    save(model3, model4, name+"v2", val_loader, DEVICE)
    free_data(train_dataset, train_loader, val_dataset, val_loader)
    return losses1, losses2, acces1, acces2


def train4():
    name = "type4"
    model1 = CnnM(input_channel=10, hidden_dim=300).to(DEVICE)
    model2 = MLP(input_dim=300, output_dim=10).to(DEVICE)
    model1.load_state_dict(torch.load('type2_model1_best.pth', map_location=DEVICE))
    model2.load_state_dict(torch.load('type2_model2_best.pth', map_location=DEVICE))
    models = [model1, model2]
    train_dataset = HandwrittenDigitsDataset('../processed_data/train', transform=ShuffleChannels(seed=42))
    val_dataset = HandwrittenDigitsDataset("../processed_data/val", transform=ShuffleChannels(seed=42))
    train_loader = get_two_class_loaders(models, train_dataset, DEVICE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    all_correct_datasets, all_wrong_datasets = get_all_class_loaders(models, train_loader, DEVICE)
    loaders = []
    for i in range(all_wrong_datasets.__len__()):
        loaders.append(all_correct_datasets[i])
        loaders.append(all_wrong_datasets[i])
    model1 = train_presentation(loaders, model1, optimizer, DEVICE, NUM_EPOCHS)
    loaders = get_clusters(model1, train_loader, DEVICE)
    losses, acces, model1, model2 = dro_train_process(loaders, models, optimizer, NUM_EPOCHS)
    save(model1, model2, name + str(i), val_loader, DEVICE)
    free_data(train_dataset, train_loader, val_dataset, val_loader)


if __name__ == "__main__":
    with open('out.txt', 'a') as f:
        tmp = sys.stdout
        sys.stdout = f
        print("start")
        # loss1, acc1 = train1()
        # print("finish train1")
        # loss2, acc2 = train2()
        print("finish train2")
        # draw_two(loss1, loss2, "BASE MODELS LOSS", "BASE MODEL1", "BASE MODEL2")
        # draw_two(acc1, acc2, "BASE MODELS ACCURACY", "BASE MODEL1", "BASE MODEL2")
        losses1, losses2, acces1, acces2 = train3()
        print("finish train3")
        draw_two(losses1, losses2, "BASE DRO MODELS LOSS", "BASE DRO MODEL1", "BASE DRO MODEL2")
        draw_two(acces1, acces2, "BASE DRO MODELS ACCURACY", "BASE DRO MODEL1", "BASE DRO MODEL2")
        # draw_three(loss2, losses1, losses2, "COMPARE MODELS LOSS", "BASE MODEL2", "BASE DRO MODEL1", "BASE DRO MODEL2")
        # draw_three(acc2, acces1, acces2, "COMPARE MODELS ACCURACY", "BASE MODEL2", "BASE DRO MODEL1", "BASE DRO MODEL2")
        sys.stdout = tmp
    # 浮点数字典转换成json兼容的数据结构（纯Python列表）
    float_dicts_json_compatible = {
        # 'loss1': loss1,
        # 'loss2': loss2,
        'losses1': losses1,
        'losses2': losses2,
        "acces1": acces1,
        "acces2": acces2

    }
    with open('list_data.json', 'w') as f:
        json.dump(float_dicts_json_compatible, f)
