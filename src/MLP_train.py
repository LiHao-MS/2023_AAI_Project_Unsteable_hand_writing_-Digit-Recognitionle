import json
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import HandwrittenDigitsDataset
import torch
import torch.nn as nn
from FULLY_MLP import MLP

BATCH_SIZE = 500  # 根据需要调整batch大小
# model_path = './models/mlp_model.pth'
num_hidden = 32
num_classes = 10
epochs = 100
lr_init = 0.001

class To1DTransform(object):
    def __call__(self, tensor):
        # 将(10, 28, 28)形状的Tensor展平为(10, 784)
        return tensor.view(tensor.shape[0], -1)

def eval(loader, model, criterion, device=None):
    '''
        Evaluate the model.
        @param loader: data loader
        @param model: model
        @param criterion: loss function
        @param device: device
    '''
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if device is not None:
                data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            total += target.size(0)
            correct += torch.sum((torch.argmax(output, dim=1) == target).float()).item()

    return {
        'loss': total_loss / (batch_idx + 1),
        'accuracy': correct / total,
    }

def train_epoch(loader, model, criterion, optimizer, device=None):
    '''
        Train the model for one epoch.
        @param loader: data loader
        @param model: model
        @param criterion: loss function
        @param optimizer: optimizer
        @param device: device
    '''
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(loader):
        if device is not None:
            data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += target.size(0)
        correct += torch.sum((torch.argmax(output, dim=1) == target).float()).item()

    return {
        'loss': total_loss / (batch_idx + 1),
        'accuracy': correct / total,
    }
def save_result(file_path,losses, acc):
    result = {
        'losses': losses,
        'acc': acc
    }
    with open(file_path, 'w') as f:
        json.dump(result, f)

def draw_pic(fiel_path, losses, acc):
    plt.figure()
    plt.plot(losses)
    plt.title('loss')
    plt.savefig(fiel_path+'mlp_loss.png')
    plt.figure()
    plt.plot(acc)
    plt.title('acc')
    plt.savefig(fiel_path+'mlp_acc.png')
    plt.cla()


def train_MLP():
    train_path = "./local/train/"
    test_path = "./local/test/"
    model_path = './local/mlp_model.pth'
    model = MLP(10 * 28 * 28, num_hidden, num_classes)
    devices = [x for x in range(torch.cuda.device_count())]
    model = torch.nn.DataParallel(model, devices)
    device = torch.device(model.device_ids[0])
    model.to(device)
    train_dataset = HandwrittenDigitsDataset('../processed_data/train', transform=None)
    val_dataset = HandwrittenDigitsDataset("../processed_data/val", transform=None)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    # losses, acc = train_epoch(train_loader, model, criterion, optimizer, device=device)
    test_loss = []
    test_acc = []
    train_loss = []
    train_acc = []
    for epoch in range(1, epochs + 1):
        train_res = train_epoch(train_loader, model, criterion, optimizer, device=device)
        test_res = eval(val_loader, model, criterion, device=device)
        test_loss.append(test_res['loss'])
        test_acc.append(test_res['accuracy'])
        train_loss.append(train_res['loss'])
        train_acc.append(train_res['accuracy'])
        print('Epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'.format(
            epoch, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy']))
    # 训练完成后，保存模型
    torch.save(model.module.state_dict(), model_path)
    draw_pic(train_path, train_loss, train_acc)
    draw_pic(test_path, test_loss, test_acc)
    save_result(train_path+'mlp_result.json', train_loss, train_acc)
    save_result(test_path+'mlp_result.json', test_loss, test_acc)

if __name__ == '__main__':
    train_MLP()


