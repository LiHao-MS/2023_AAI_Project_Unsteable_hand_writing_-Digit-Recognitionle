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
    """
       This function takes a list of models, an input dataloader, and the device for computations. It separates
       the dataset into correct and wrong predictions per class, creating separate dataloaders for each category.

       Args:
       - models (list): A list containing two models in evaluation mode.
       - dataloader (DataLoader): The dataloader providing batches of data.
       - DEVICE (torch.device): Device identifier on which the models will run.

       Returns:
       - all_correct_datasets (dict): Dictionary with dataloaders for correctly classified samples, indexed by class labels.
       - all_wrong_datasets (dict): Dictionary with dataloaders for incorrectly classified samples, indexed by class labels.
    """

    # Move both models to evaluation mode
    model1 = models[0].eval()
    model2 = models[1].eval()
    model = lambda x: model2(model1(x))

    # Initialize lists to store correct and wrong samples along with their labels
    correct_datasets = defaultdict(list)
    wrong_datasets = defaultdict(list)
    correct_labels = defaultdict(list)
    wrong_labels = defaultdict(list)

    # Iterate over the data without gradients
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            predict = torch.argmax(outputs, dim=1)

            # Iterate over the batch
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

    # Create new datasets and loaders for the correct and wrong samples
    for label_ in correct_datasets.keys():
        correct_data = torch.stack(correct_datasets[label_], dim=0)
        wrong_data = torch.stack(wrong_datasets[label_], dim=0)

        # Create new datasets and loaders for the correct and wrong samples
        correct_dataset = TensorDataset(correct_data, torch.tensor(correct_labels[label_]))
        wrong_dataset = TensorDataset(wrong_data, torch.tensor(wrong_labels[label_]))

        # Create new dataloaders for the correct and wrong samples
        correct_dataloader = DataLoader(correct_dataset, batch_size=dataloader.batch_size, shuffle=True)
        wrong_dataloader = DataLoader(wrong_dataset, batch_size=dataloader.batch_size, shuffle=True)

        # Store the dataloaders in dictionaries
        all_correct_datasets[label_] = correct_dataloader
        all_wrong_datasets[label_] = wrong_dataloader
    return all_correct_datasets, all_wrong_datasets


def get_two_class_loaders(models, dataloader, DEVICE,BATCH_SIZE):
    """
        This function takes in two models from a list, a dataloader, the device (CPU or GPU), and the batch size.
        It evaluates both models and combines them into a single model pipeline. It then separates the data
        based on correct and incorrect predictions made by the combined model, storing each group in separate lists.
        Finally, it creates new TensorDatasets and DataLoader instances for correctly classified and incorrectly classified data.

        Args:
            models (List[torch.nn.Module]): A list containing two PyTorch models in evaluation mode.
            dataloader (torch.utils.data.DataLoader): The original data loader providing batches of data.
            DEVICE (torch.device): The device to which tensors should be transferred.
            BATCH_SIZE (int): The batch size to use for the new loaders.

        Returns:
            Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: A tuple containing two DataLoader instances,
                one for correctly classified data and another for incorrectly classified data.
    """

    # Evaluate the models and create a lambda function to chain their operations
    model1 = models[0].eval()
    model2 = models[1].eval()
    model = lambda x: model2(model1(x))

    # Initialize lists to store correct and wrong samples along with their labels
    correct_data = []
    correct_data_label = []
    wrong_data = []
    wrong_data_label = []

    # Iterate over the data without gradients
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            predict = torch.argmax(outputs, dim=1)
            correct = target == predict
            correct_data.append(data[correct])
            correct_data_label.append(target[correct])
            wrong_data.append(data[~correct])
            wrong_data_label.append(target[~correct])

    # Concatenate all tensors into a single tensor
    correct_data = torch.cat(correct_data, dim=0).cpu()
    correct_data_label = torch.cat(correct_data_label, dim=0).cpu()
    wrong_data = torch.cat(wrong_data, dim=0).cpu()
    wrong_data_label = torch.cat(wrong_data_label, dim=0).cpu()

    # Create new datasets and loaders for the correct and wrong samples
    correct_dataset = TensorDataset(correct_data, correct_data_label)
    wrong_dataset = TensorDataset(wrong_data, wrong_data_label)

    # Create new dataloaders for the correct and wrong samples
    correct_dataloader = DataLoader(correct_dataset, batch_size=BATCH_SIZE, shuffle=True)
    wrong_dataloader = DataLoader(wrong_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return correct_dataloader, wrong_dataloader

def train_presentation(train_loaders, model, opt, DEVICE, NUM_EPOCHS):
    """
        This function takes in a list of dataloaders, a model, an optimizer, a device, and the number of epochs.
        It trains the model on the data provided by the dataloaders for the specified number of epochs.

        Args:
            train_loaders (List[torch.utils.data.DataLoader]): A list of dataloaders providing batches of data.
            model (torch.nn.Module): The model to train.
            opt (torch.optim.Optimizer): The optimizer to use for training.
            DEVICE (torch.device): The device to which tensors should be transferred.
            NUM_EPOCHS (int): The number of epochs for which to train the model.

        Returns:
            torch.nn.Module: The trained model.
    """

    n_group_pairs = len(train_loaders)
    model.train()

    # Iterate over the epochs
    for epoch in range(NUM_EPOCHS):
        for batches in zip(*train_loaders):
            # work on each batch
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
    """
        This function takes in a model, a dataloader, and a device. It evaluates the model on the data provided by the dataloader
        and clusters the data based on the model's predictions. It then creates new dataloaders for each cluster of data.

        Args:
            model (torch.nn.Module): The model to use for clustering.
            dataloader (torch.utils.data.DataLoader): The dataloader providing batches of data.
            DEVICE (torch.device): The device to which tensors should be transferred.

        Returns:
            List[torch.utils.data.DataLoader]: A list of dataloaders, one for each cluster of data.
    """
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

    # cluster the data
    for k, v in groups.items():
        x = np.stack(v["encoder"], axis=0)
        cur_clusters = {}
        centroid, label, inertia = k_means(x, 2)
        # get the data for each cluster
        for cluster_id, data in zip(label, v["data"]):
            if cluster_id not in cur_clusters:
                cur_clusters[cluster_id] = []
            cur_clusters[cluster_id].append(data)

        # create new dataloaders for each cluster
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
    """
        This function takes in a list of dataloaders, a list of models, an optimizer, a device, and the number of epochs.
        It trains the models on the data provided by the dataloaders for the specified number of epochs.

        Args:
            train_loaders (List[torch.utils.data.DataLoader]): A list of dataloaders providing batches of data.
            models (List[torch.nn.Module]): A list of models to train.
            opt (torch.optim.Optimizer): The optimizer to use for training.
            DEVICE (torch.device): The device to which tensors should be transferred.
            NUM_EPOCHS (int): The number of epochs for which to train the models.

        Returns:
            Tuple[List[float], List[float], torch.nn.Module, torch.nn.Module]: A tuple containing two lists of floats,
                one for the losses and one for the accuracies, the trained models.
    """
    losses = []
    acces = []
    model1 = models[0]
    model2 = models[1]

   # Iterate over the epochs
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

            # get the worst loss
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


