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
NUM_EPOCHS = 60 # 根据需要调整epoch数量
DRO_NUM_EPOCH = 400
WEIGHT_DECAY = 0.001  # 学习率衰减
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_EVERY = 10
HIDDEN_DIM = 300

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
    return losses, acc

def train2():
    name = "type2"
    model1 = CnnM(input_channel=10, hidden_dim=HIDDEN_DIM).to(DEVICE)
    model2 = MLP(input_dim=HIDDEN_DIM, output_dim=10).to(DEVICE)
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
    return losses, acc

def train_fake():
    name = "type_fake"
    model1 = CnnM(input_channel=10, hidden_dim=HIDDEN_DIM).to(DEVICE)
    model2 = MLP(input_dim=HIDDEN_DIM, output_dim=10).to(DEVICE)
    train_dataset = HandwrittenDigitsDataset('../processed_data/train', transform=IdentityTransform())
    val_dataset = HandwrittenDigitsDataset("../processed_data/val",transform=IdentityTransform())
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
    return losses, acc

def train3():
    name = "type3"
    model1 = CnnM(input_channel=10, hidden_dim=300).to(DEVICE)
    model2 = MLP(input_dim=300, output_dim=10).to(DEVICE)
    model1.load_state_dict(torch.load('./models/type_fake_final_model1.pth', map_location=DEVICE))
    model2.load_state_dict(torch.load('./models/type_fake_final_model2.pth', map_location=DEVICE))
    train_dataset = HandwrittenDigitsDataset('../processed_data/train', transform=IdentityTransform())
    val_dataset = HandwrittenDigitsDataset("../processed_data/val", transform=IdentityTransform())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    models = [model1, model2]
    loaders = get_two_class_loaders(models, train_loader, DEVICE, BATCH_SIZE)
    losses1, acces1, model1, model2 = dro_train_process(loaders, models, optimizer, DEVICE, DRO_NUM_EPOCH)
    save(model1, model2, name + "v1", val_loader, DEVICE)
    del optimizer

    model3 = CnnM(input_channel=10, hidden_dim=300).to(DEVICE)
    model4 = MLP(input_dim=300, output_dim=10).to(DEVICE)
    models = [model3, model4]
    optimizer2 = torch.optim.Adam(list(model3.parameters()) + list(model4.parameters()), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    losses2, acces2, model3, model4 = dro_train_process(loaders, models, optimizer2, DEVICE, DRO_NUM_EPOCH)
    save(model3, model4, name+"v2", val_loader, DEVICE)
    return losses1, losses2, acces1, acces2


def train4():
    name = "type4"
    model1 = CnnM(input_channel=10, hidden_dim=HIDDEN_DIM).to(DEVICE)
    model2 = MLP(input_dim=HIDDEN_DIM, output_dim=10).to(DEVICE)
    model1.load_state_dict(torch.load('./models/type_fake_final_model1.pth', map_location=DEVICE))
    model2.load_state_dict(torch.load('./models/type_fake_final_model2.pth', map_location=DEVICE))
    models = [model1, model2]
    train_dataset = HandwrittenDigitsDataset('../processed_data/train', transform=IdentityTransform())
    val_dataset = HandwrittenDigitsDataset("../processed_data/val", transform=IdentityTransform())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_correct_datasets, all_wrong_datasets = get_all_class_loaders(models, train_loader, DEVICE)
    loaders = []
    for i in all_wrong_datasets.keys():
        if min(len(all_correct_datasets[i]), len(all_wrong_datasets[i])) < 5:
            continue
        loaders.append(all_correct_datasets[i])
        loaders.append(all_wrong_datasets[i])
    print("train4: all loaders {}".format(loaders))

    optimizer2 = torch.optim.Adam(list(model1.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model3 = train_presentation(loaders, model1, optimizer2, DEVICE, NUM_EPOCHS)
    
    del all_correct_datasets, all_wrong_datasets, loaders
    
    loaders = get_clusters(model3, train_loader, DEVICE)
    
    del model3
    
    model1.load_state_dict(torch.load('./models/type_fake_final_model1.pth', map_location=DEVICE))
    optimizer3 = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    models = [model1, model2]
    losses1, acces1, model1, model2 = dro_train_process(loaders, models, optimizer3, DEVICE, DRO_NUM_EPOCH)
    save(model1, model2, name + "v1", val_loader, DEVICE)
    del optimizer3

    model4 = CnnM(input_channel=10, hidden_dim=300).to(DEVICE)
    model5 = MLP(input_dim=300, output_dim=10).to(DEVICE)
    optimizer4 = torch.optim.Adam(list(model4.parameters()) + list(model5.parameters()), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    models = [model4, model5]
    losses2, acces2, model4, model5 = dro_train_process(loaders, models, optimizer4, DEVICE, DRO_NUM_EPOCH)

    save(model4, model5, name + "v2", val_loader, DEVICE)
    return losses1, losses2, acces1, acces2


def train_first(file_name):
    print("start train first")
    with open('{}.json'.format(file_name), 'a') as f:
        loss1, acc1 = train1()
        print("finish train1")

        loss2, acc2 = train2()
        print("finish train2")

        loss3, acc3 = train_fake()
        print("finish train_fake")
        float_dicts_json_compatible = {
            'loss1': loss1,
            'acc1': acc1,
            'loss2': loss2,
            'acc2': acc2,
            'loss_fake': loss3,
            'acc_fake': acc3,
        }
        json.dump(float_dicts_json_compatible, f)
        f.write('\n')
    return loss1, loss2, loss3, acc1, acc2, acc3

def train_last(file):
    print("start train last")
    with open('{}.json'.format(file), 'a') as f:
        losses1, losses2, acces1, acces2 = train3()
        print("finish train3")
        losses3, losses4, acces3, acces4 = train4()
        print("finish train4")
        float_dicts_json_compatible = {
            'losses1': losses1,
            'losses2': losses2,
            'acces1': acces1,
            'acces2': acces2,
            'losses3': losses3,
            'losses4': losses4,
            'acces3': acces3,
            'acces4': acces4,
        }
        json.dump(float_dicts_json_compatible, f)
        f.write('\n')
    return losses1, losses2, losses3, losses4, acces1, acces2, acces3, acces4

def res_show():
    with open("./json/LossAndAcc.json", 'r') as f:
       record = json.load(f)
    loss1 = record['loss1']
    loss2 = record['loss2']
    loss3 = record['loss_fake']
    acc1 = record['acc1']
    acc2 = record['acc2']
    acc3 = record['acc_fake']
    losses1 = [i*100 for i in record['losses1']]
    losses2 = [i*100 for i in record['losses2']]
    losses3 = [i*100 for i in record['losses3']]
    losses4 = [i*100 for i in record['losses4']]
    acces1 = [i*100 for i in record['acces1']]
    acces2 = [i*100 for i in record['acces2']]
    acces3 = [i*100 for i in record['acces3']]
    acces4 = [i*100 for i in record['acces4']]
    losses = [loss1, loss2, loss3, losses1, losses2, losses3, losses4]
    accs = [acc1, acc2, acc3, acces1, acces2, acces3, acces4]
    draw_all(losses, accs)

if __name__ == "__main__":
    res_show()
    file_name = "./json/LossAndAcc"
    # # loss1, loss2, loss3, acc1, acc2, acc3 = train_first(file_name)
    # losses1, losses2, losses3, losses4, acces1, acces2, acces3, acces4 = train_last(file_name)
    # with open("./json/LossAndAcc.json", 'r') as f:
    #    record = json.load(f)
    # loss1 = record['loss1']
    # loss2 = record['loss2']
    # loss3 = record['loss_fake']
    # acc1 = record['acc1']
    # acc2 = record['acc2']
    # acc3 = record['acc_fake']
    # losses = [loss1, loss2, loss3, losses1, losses2, losses3, losses4]
    # accs = [acc1, acc2, acc3, acces1, acces2, acces3, acces4]
    # draw_all(losses, accs)
