from dataset import HandwrittenDigitsDataset
from CNN_model import CnnM
from mlp_model import MLP
import json
from utils import *
from DRO import *
from torch.utils.data import DataLoader
from MLP_train import train_MLP

BATCH_SIZE = 100  # 根据需要调整batch大小
LEARNING_RATE = 0.001  # 学习率
NUM_EPOCHS = 100 # 根据需要调整epoch数量
DRO_NUM_EPOCH = 200
WEIGHT_DECAY = 0.01  # 学习率衰减
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_EVERY = 10
HIDDEN_DIM = 300


def train_fake():
    name = "odd_fake"
    model1 = CnnM(input_channel=10, hidden_dim=HIDDEN_DIM).to(DEVICE)
    model2 = MLP(input_dim=HIDDEN_DIM, output_dim=10).to(DEVICE)
    odd_train_dataset = HandwrittenDigitsDataset('../processed_data/train', transform=IdentityTransform(), odd=True)
    val_dataset = HandwrittenDigitsDataset("../processed_data/val",transform=IdentityTransform(), odd=True)
    train_loader = DataLoader(odd_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    criterion = F.cross_entropy
    losses, acc, model1, model2 = base_train_process(NUM_EPOCHS, train_loader, DEVICE, optimizer, model2, model1,
                                                     criterion,
                                                     SAVE_EVERY, val_loader, name)
    # 训练完成后，保存模型
    save(model1, model2, name, val_loader, DEVICE)
    return losses, acc

def train_even():
    name = "even_fake"
    model1 = CnnM(input_channel=10, hidden_dim=HIDDEN_DIM).to(DEVICE)
    model2 = MLP(input_dim=HIDDEN_DIM, output_dim=10).to(DEVICE)
    model1.load_state_dict(torch.load('./models/odd_fake_final_model1.pth', map_location=DEVICE))
    model2.load_state_dict(torch.load('./models/odd_fake_final_model2.pth', map_location=DEVICE))
    models = [model1, model2]
    train_dataset = HandwrittenDigitsDataset('../processed_data/train', transform=IdentityTransform(), even=True)
    val_dataset = HandwrittenDigitsDataset("../processed_data/val", transform=IdentityTransform(), even=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_correct_datasets, all_wrong_datasets = get_all_class_loaders(models, train_loader, DEVICE)
    loaders = []
    for i in all_wrong_datasets.keys():
        loaders.append(all_correct_datasets[i])
        loaders.append(all_wrong_datasets[i])
        
    optimizer2 = torch.optim.Adam(list(model1.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model3 = train_presentation(loaders, model1, optimizer2, DEVICE, NUM_EPOCHS)
    
    del all_correct_datasets, all_wrong_datasets, loaders
    
    loaders = get_clusters(model3, val_loader, DEVICE)
    
    del model3
    
    model1.load_state_dict(torch.load('./models/odd_fake_final_model1.pth', map_location=DEVICE))
    optimizer3 = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    models = [model1, model2]
    losses1, acces1, model1, model2 = dro_train_process(loaders, models, optimizer3, DEVICE, DRO_NUM_EPOCH)
    save(model1, model2, name, val_loader, DEVICE)
    del optimizer3
    return losses1, acces1,

def train_target():
    name = "even_to_all"
    model1 = CnnM(input_channel=10, hidden_dim=HIDDEN_DIM).to(DEVICE)
    model2 = MLP(input_dim=HIDDEN_DIM, output_dim=10).to(DEVICE)
    model1.load_state_dict(torch.load('./models/even_fake_final_model1.pth', map_location=DEVICE))
    model2.load_state_dict(torch.load('./models/even_fake_final_model2.pth', map_location=DEVICE))
    train_dataset = HandwrittenDigitsDataset('../processed_data/train', transform=IdentityTransform())
    val_dataset = HandwrittenDigitsDataset("../processed_data/val", transform=IdentityTransform())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    criterion = F.cross_entropy
    losses, acc, model1, model2 = base_train_process(NUM_EPOCHS, train_loader, DEVICE, optimizer, model2, model1,
                                                     criterion,
                                                     SAVE_EVERY, val_loader, name)
    save(model1, model2, name, val_loader, DEVICE)
    return losses, acc

def train_last(file):
    print("start train last")
    with open('{}.json'.format(file), 'a') as f:
        loss1, acc1 = train_fake()
        print("finish train fake")
        loss2, acc2 = train_even()
        print("finish train even")
        loss3, acc3 = train_target()
        print("finish train4")
        float_dicts_json_compatible = {
            'loss1': loss1,
            'loss2': loss2,
            'acc1': acc1,
            'acc2': acc2,
            'loss3': loss3,
            'acc3': acc3,
        }
        json.dump(float_dicts_json_compatible, f)
        f.write('\n')
    return loss1, loss2, loss3, acc1, acc2, acc3

if __name__ == "__main__":
    file_name = "./json/odd_LossAndAcc"

    loss1, loss2, loss3, acc1, acc2, acc3 = train_last(file_name)

    losses = [loss1, loss2, loss3]
    accs = [acc1, acc2, acc3]
    draw_odd(losses, accs)
