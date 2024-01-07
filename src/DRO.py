import torch
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
from utils import compute_l2
from sklearn import metrics
from sklearn.cluster import k_means
import numpy as np
import torch.nn.functional as F
from collections import Counter

def get_all_class_loaders(models, dataloader, DEVICE):
    model1 = models[0].eval()
    model2 = models[1].eval()
    model = lambda x: model2(model1(x))
    correct_datasets = defaultdict(list)
    wrong_datasets = defaultdict(list)
    correct_labels = defaultdict(list)
    wrong_labels = defaultdict(list)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(DEVICE), target.to(DEVICE)  # 将数据转移到正确的设备上
            outputs = model(data)
            predict = torch.argmax(outputs, dim=1)
            for i in range(len(target)):
                label = target[i]
                pred = predict[i]
                if label == pred:
                    key = int(label.cpu().int().item())
                    correct_datasets[key].append(data[i])
                    correct_labels[key].append(label)
                else:
                    key = int(label.cpu().int().item())
                    wrong_datasets[key].append(data[i])
                    wrong_labels[key].append(label)

    all_correct_datasets = {}
    all_wrong_datasets = {}
    for label_ in correct_datasets.keys():
        correct_data = torch.stack(correct_datasets[label_], dim=0)
        wrong_data = torch.stack(wrong_datasets[label_], dim=0)

        correct_dataset = TensorDataset(correct_data, torch.tensor(correct_labels[label_]))
        wrong_dataset = TensorDataset(wrong_data, torch.tensor(wrong_labels[label_]))

        correct_dataloader = DataLoader(correct_dataset, batch_size=dataloader.batch_size, shuffle=True)
        wrong_dataloader = DataLoader(wrong_dataset, batch_size=dataloader.batch_size, shuffle=True)

        all_correct_datasets[label_] = correct_dataloader
        all_wrong_datasets[label_] = wrong_dataloader
    return all_correct_datasets, all_wrong_datasets


def get_two_class_loaders(models, dataloader, DEVICE,BATCH_SIZE):
    model1 = models[0].eval()
    model2 = models[1].eval()
    model = lambda x: model2(model1(x))
    correct_data = []
    correct_data_label = []
    wrong_data = []
    wrong_data_label = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(DEVICE), target.to(DEVICE)  # 将数据转移到正确的设备上
            outputs = model(data)
            predict = torch.argmax(outputs, dim=1)
            correct = target == predict
            correct_data.append(data[correct])
            correct_data_label.append(target[correct])
            wrong_data.append(data[~correct])
            wrong_data_label.append(target[~correct])
    # 因为每次迭代返回的是一个批次的数据，最后需要将所有批次的结果合并
    correct_data = torch.cat(correct_data, dim=0).cpu()
    correct_data_label = torch.cat(correct_data_label, dim=0).cpu()
    wrong_data = torch.cat(wrong_data, dim=0).cpu()
    wrong_data_label = torch.cat(wrong_data_label, dim=0).cpu()

    # 创建新的TensorDataset
    correct_dataset = TensorDataset(correct_data, correct_data_label)
    wrong_dataset = TensorDataset(wrong_data, wrong_data_label)

    # 创建新的DataLoader
    correct_dataloader = DataLoader(correct_dataset, batch_size=BATCH_SIZE, shuffle=True)
    wrong_dataloader = DataLoader(wrong_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return correct_dataloader, wrong_dataloader

def train_presentation(train_loaders, model, opt, DEVICE, NUM_EPOCHS):
    n_group_pairs = len(train_loaders)
    model.train()
    for epoch in range(NUM_EPOCHS):  # 训练循环
        for batches in zip(*train_loaders):
            for i in range(n_group_pairs//2):
                pos_data, pos_label = batches[i*2]
                neg_data, neg_label = batches[i * 2+1]
                pos_data, neg_data = pos_data.to(DEVICE), neg_data.to(DEVICE)

                min_size = min(len(pos_data), len(neg_data))
                pos_data = pos_data[:min_size]
                neg_data = neg_data[:min_size]

                pos_encoder = model(pos_data)
                neg_encoder = model(neg_data)

                diff_pos_pos = compute_l2(pos_encoder, pos_encoder)
                diff_pos_neg = compute_l2(pos_encoder, neg_encoder)

                loss = (
                    torch.mean(torch.max(torch.zeros_like(diff_pos_pos),
                                        diff_pos_pos - diff_pos_neg +
                                        torch.ones_like(diff_pos_pos) * 0.3)))
                loss.backward()
            opt.step()
            opt.zero_grad()
    return model


def get_clusters(model, dataloader, DEVICE):
    model.eval()
    groups = {}
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)  # 前向传播
            x_s = output.detach().cpu().numpy()
            y_s = target.detach().cpu().numpy()
            for x, y, da in zip(x_s, y_s, data):
                if int(y) not in groups:
                    groups[int(y)] = {"encoder":[], "data":[]}
                groups[int(y)]["encoder"].append(x)
                groups[int(y)]["data"].append(da)
    dataloaders = []
    for k, v in groups.items():
        x = np.stack(v["encoder"], axis=0)
        cur_clusters = {}
        centroid, label, inertia = k_means(x, 2)
        for cluster_id, data in zip(label, v["data"]):
            if cluster_id not in cur_clusters:
                cur_clusters[cluster_id] = []
            cur_clusters[cluster_id].append(data)

        for key in cur_clusters.keys():
            data_ = torch.stack(cur_clusters[key], dim=0)
            target_ = torch.ones(len(cur_clusters[key])) * k
            dataset_ = TensorDataset(data_, target_)
            dataloader_ = DataLoader(dataset_, batch_size=dataloader.batch_size, shuffle=True)
            dataloaders.append(dataloader_)
            print("cluster {},label:{}, length:{}".format(k, key, len(data_)))
    print("total cluster:{}".format(len(dataloaders)))
    return dataloaders


def dro_train_process(train_loaders, models, opt, DEVICE, NUM_EPOCHS):
    losses = []
    acces = []
    model1 = models[0]
    model2 = models[1]
    # n_group_pairs = len(train_loaders)
    for num in range(NUM_EPOCHS):
        cur_loss = 0
        cur_acc = 0
        cur_len = 0
        model1.train()
        model2.train()

        # every two loaders are paired together
        for batches in zip(*train_loaders):
            # work on each batch
            worse_acc = 0
            worst_len = 0
            x, y = [], []
            for batche in batches:
                data, target = batche
                x.append(data.to(DEVICE))
                y.append(target.to(DEVICE))

            pred = model2(model1(torch.cat(x, dim=0)))
            worst_loss = 0
            cur_idx = 0
            for cur_true in y:
                cur_pred = pred[cur_idx:cur_idx+cur_true.shape[0]]
                cur_idx += len(cur_true)
                loss = F.cross_entropy(cur_pred, cur_true.long())
                if loss.item() > worst_loss:
                    worst_loss = loss
                    worst_len = len(cur_true)
                    worse_acc = torch.sum((torch.argmax(cur_pred, dim=1) == cur_true).float()).item()
            cur_loss += worst_loss.item()
            cur_acc += worse_acc
            cur_len += worst_len
            opt.zero_grad()
            worst_loss.backward()
            opt.step()
        losses.append(cur_loss / cur_len)
        acces.append(cur_acc / cur_len)

    return losses, acces, model1, model2


