from dataset import HandwrittenDigitsDataset
from CNN_model import CnnM
from mlp_model import MLP
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from utils import *
BATCH_SIZE = 64
LEARNING_RATE = 0.001 # 学习率
NUM_EPOCHS = 500 # 根据需要调整epoch数量
WEIGHT_DECAY = 0.01 # 学习率衰减
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
    optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = F.cross_entropy
    losses, model1, model2 = base_train_process(NUM_EPOCHS, train_loader, DEVICE, optimizer, model2, model1, criterion, SAVE_EVERY, val_loader, name)
    #训练完成后，保存模型
    save_drae(model1, model2, losses, name, val_loader, DEVICE)
    free_data(train_dataset, train_loader, val_dataset, val_loader)

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
    losses, model1, model2 = base_train_process(NUM_EPOCHS, train_loader, DEVICE, optimizer, model2, model1, criterion,
                                                SAVE_EVERY, val_loader,name)
    # 训练完成后，保存模型
    save_drae(model1, model2, losses, name, val_loader, DEVICE)
    free_data(train_dataset, train_loader, val_dataset, val_loader)

def train3():
    name = "type3"
    model1 = CnnM(input_channel=10, hidden_dim=300).to(DEVICE)
    model2 = MLP(input_dim=300, output_dim=10).to(DEVICE)
    print("dro")

if __name__ == "__main__":
    print("start")
    train1()
    print("finish train1")
    train2()
    print("finish train2")