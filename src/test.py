from dataset import HandwrittenDigitsDataset
from CNN_model import CnnM
from mlp_model import MLP
import torch
from torch.utils.data import DataLoader, TensorDataset
from test_dataset import TestDataset
from utils import IdentityTransform

BATCH_SIZE = 100  # 根据需要调整batch大小
LEARNING_RATE = 0.001  # 学习率
NUM_EPOCHS = 60 # 根据需要调整epoch数量
DRO_NUM_EPOCH = 400
WEIGHT_DECAY = 0.001  # 学习率衰减
HIDDEN_DIM = 300

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1 = CnnM(input_channel=10, hidden_dim=HIDDEN_DIM).to(DEVICE)
model2 = MLP(input_dim=HIDDEN_DIM, output_dim=10).to(DEVICE)
# train_dataset = HandwrittenDigitsDataset('../processed_data/train', transform=IdentityTransform())
# val_dataset = HandwrittenDigitsDataset("../processed_data/val", transform=IdentityTransform())
# test_dataset = TestDataset('../processed_data/test')
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
# test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE, shuffle=False)

def test_acc(model_path, loader):
    model1.load_state_dict(torch.load(model_path+'1.pth', map_location=DEVICE))
    model2.load_state_dict(torch.load(model_path+'2.pth', map_location=DEVICE))
    model1.eval()
    model2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):  # 迭代加载数据
            data, target = data.to(DEVICE), target.to(DEVICE)  # 将数据转移到正确的设备上
            outputs = model2(model1(data))
            correct += torch.sum((torch.argmax(outputs, dim=1) == target).float()).item()
            total += target.size(0)  # 累加总样本数
    return correct / total

def get_acc_loss():
    train_dataset = HandwrittenDigitsDataset('../processed_data/train', transform=IdentityTransform())
    val_dataset = HandwrittenDigitsDataset("../processed_data/val", transform=IdentityTransform())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    fake_model_path = './models/type_fake_final_model'
    type3v1_model_path = './models/type3v1_final_model'
    type3v2_model_path = './models/type3v2_final_model'
    type4v1_model_path = './models/type4v1_final_model'
    type4v2_model_path = './models/type4v2_final_model'
    fake_acc_train = test_acc(fake_model_path, train_loader)
    type3v1_acc_train = test_acc(type3v1_model_path, train_loader)
    type3v2_acc_train = test_acc(type3v2_model_path, train_loader)
    type4v1_acc_train = test_acc(type4v1_model_path, train_loader)
    type4v2_acc_train = test_acc(type4v2_model_path, train_loader)
    print('fake_acc_train:', fake_acc_train)
    print('type3v1_acc_train:', type3v1_acc_train)
    print('type3v2_acc_train:', type3v2_acc_train)
    print('type4v1_acc_train:', type4v1_acc_train)
    print('type4v2_acc_train:', type4v2_acc_train)
    fake_acc_val = test_acc(fake_model_path, val_loader)
    type3v1_acc_val = test_acc(type3v1_model_path, val_loader)
    type3v2_acc_val = test_acc(type3v2_model_path, val_loader)
    type4v1_acc_val = test_acc(type4v1_model_path, val_loader)
    type4v2_acc_val = test_acc(type4v2_model_path, val_loader)
    print('fake_acc_val:', fake_acc_val)
    print('type3v1_acc_val:', type3v1_acc_val)
    print('type3v2_acc_val:', type3v2_acc_val)
    print('type4v1_acc_val:', type4v1_acc_val)
    print('type4v2_acc_val:', type4v2_acc_val)


def test_res():
    model_path = './models/type4v2_final_model'
    test_dataset = TestDataset('../processed_data/test')
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE, shuffle=False)
    model1.load_state_dict(torch.load(model_path+'1.pth', map_location=DEVICE))
    model2.load_state_dict(torch.load(model_path+'2.pth', map_location=DEVICE))
    model1.eval()
    model2.eval()
    correct = 0
    total = 0
    with open("./res.txt", 'w') as f:
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):  # 迭代加载数据
                data = data.to(DEVICE) # 将数据转移到正确的设备上
                outputs = model2(model1(data))
                res = torch.argmax(outputs, dim=1)
                for i, j in zip(res, target):
                    f.write(j)
                    f.write(" ")
                    f.write(str(int(i)))
                    f.write('\n')

    return
if __name__ == "__main__":
    test_res()
