from dataset import HandwrittenDigitsDataset
from CNN_model import CnnM
from mlp_model import MLP
import json
from utils import *
from DRO import *
from torch.utils.data import DataLoader
from test_dataset import TestDataset
import argparse

# Set hyperparameters
BATCH_SIZE = 100
LEARNING_RATE = 0.001
NUM_EPOCHS = 60 # used for base models and representation training in train4()
DRO_NUM_EPOCH = 400 # used for DRO training in train3() and train4()
WEIGHT_DECAY = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_DIM = 300

def train1():
    """
    Train a model with data preprocessing: reducing dimensions to one by summing them up.

    Returns:
        losses (list): List of training losses.
        acc (list): List of validation accuracies.
    """
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
                                                     criterion)
    # save model
    save(model1, model2, name, val_loader, DEVICE)
    return losses, acc

def train2():
    """
        Train a model with data preprocessing: shuffling the first dimension.

        Returns:
            losses (list): List of training losses.
            acc (list): List of validation accuracies.
    """
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
                                                     criterion)
    # save model
    save(model1, model2, name, val_loader, DEVICE)
    return losses, acc

def train_fake():
    """
       Train fake model without data preprocessing.

       Returns:
           losses (list): List of training losses.
           acc (list): List of validation accuracies.
    """
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
                                                     criterion)

    # save model
    save(model1, model2, name, val_loader, DEVICE)
    return losses, acc

def train3():
    """
        Train two sets of models using simple DRO method.

        Returns:
            losses1, losses2: Lists of training losses for each set.
            acces1, acces2: Lists of validation accuracies for each set.
    """
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
    """
       Train two sets of models using TOFU strategy.

       Returns:
           losses1, losses2: Lists of training losses for each set.
           acces1, acces2: Lists of validation accuracies for each set.
    """
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

    # get all classes of data by fake model
    all_correct_datasets, all_wrong_datasets = get_all_class_loaders(models, train_loader, DEVICE)
    loaders = []
    for i in all_wrong_datasets.keys():
        if min(len(all_correct_datasets[i]), len(all_wrong_datasets[i])) < 5:
            continue
        loaders.append(all_correct_datasets[i])
        loaders.append(all_wrong_datasets[i])
    print("train4: all loaders {}".format(len(loaders)))

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
    """
       Train and record results for train1(), train2(), train_fake().

       Args:
           file_name (str): Name of the JSON file to store results.

       Returns:
           loss1, loss2, loss3: Training losses for each experiment.
           acc1, acc2, acc3: Validation accuracies for each experiment.
    """
    print("start train first")
    loss1, acc1 = train1()
    print("finish train1")

    loss2, acc2 = train2()
    print("finish train2")

    loss3, acc3 = train_fake()
    print("finish train_fake")

    with open('{}.json'.format(file_name), 'w') as f:
        record = json.load(f)
        float_dicts_json_compatible = {
            'loss1': loss1,
            'acc1': acc1,
            'loss2': loss2,
            'acc2': acc2,
            'loss_fake': loss3,
            'acc_fake': acc3,
        }
        json.dump(float_dicts_json_compatible.update(record), f)
        f.write('\n')
    return loss1, loss2, loss3, acc1, acc2, acc3

def train_last(file_name):
    """
       Train and record results for train3(), train4().

       Args:
           file_name (str): Name of the JSON file to store results.

       Returns:
           losses1, losses2, losses3, losses4: Training losses for each experiment.
           acces1, acces2, acces3, acces4: Validation accuracies for each experiment.
    """
    print("start train last")
    losses1, losses2, acces1, acces2 = train3()
    print("finish train3")
    losses3, losses4, acces3, acces4 = train4()
    print("finish train4")
    with open('{}.json'.format(file_name), 'w') as f:
        record = json.load(f)
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
        json.dump(float_dicts_json_compatible.update(record), f)
        f.write('\n')
    return losses1, losses2, losses3, losses4, acces1, acces2, acces3, acces4

def res_show(file_path):
    """
       This function reads a JSON file containing training metrics and prepares the data to be visualized.

       Args:
           file_path (str): The path to the JSON file that records the training losses and accuracies.

       It opens the JSON file, loads its content into a dictionary, extracts the necessary loss and accuracy values,
       scales some of them by 100, and then organizes the data into lists. Finally, it calls the `draw_all` function
       which is assumed to plot these lists representing different loss functions and accuracy measures during the training process.

       Inside the function:
       - `loss1`, `loss2`, `loss3` represent scalar final losses.
       - `acc1`, `acc2`, `acc3` represent scalar final accuracies.
       - `losses1`, `losses2`, `losses3`, `losses4` are lists of epoch-wise losses scaled by 100.
       - `acces1`, `acces2`, `acces3`, `acces4` are lists of epoch-wise accuracies scaled by 100.
    """
    with open("{}.json".format(file_path), 'r') as f:
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
    # After processing the data from the JSON file, both losses and accuracies are passed to `draw_all` for visualization.
    draw_all(losses, accs)
def test_res():
    """
       This function loads previously trained models (CnnM and MLP) from their saved state dictionaries,
       evaluates them on a testing dataset, and writes the predicted labels alongside the true labels
       to a text file named '../res.txt'.

       Inside the function:
       - `model1` is an instance of CnnM model.
       - `model2` is an instance of MLP model.
       - `test_dataset` is a TestDataset object providing testing samples.
       - `test_loader` is a DataLoader for iterating over batches of the testing dataset.
       - Models' weights are loaded from corresponding '.pth' files.
       - Predictions are made using the forward pass of both models and the argmax operation is used to obtain class labels.
    """
    model1 = CnnM(input_channel=10, hidden_dim=HIDDEN_DIM).to(DEVICE)
    model2 = MLP(input_dim=HIDDEN_DIM, output_dim=10).to(DEVICE)
    model_path = './models/type4v2_final_model'
    test_dataset = TestDataset('../processed_data/test')
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE, shuffle=False)
    model1.load_state_dict(torch.load(model_path+'1.pth', map_location=DEVICE))
    model2.load_state_dict(torch.load(model_path+'2.pth', map_location=DEVICE))
    model1.eval()
    model2.eval()
    with open("../res.txt", 'w') as f:
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

def get_parser():
    """
        This function initializes an ArgumentParser with two arguments: '--model' and '--json_name'.
        These arguments allow users to specify which action to take when running the script (e.g., test the model or visualize results).
    """
    parser = argparse.ArgumentParser(description='AAI PROJECT')
    parser.add_argument('--model', type=int, default=1)
    parser.add_argument('--json_name', type=str, default='LossAndAcc')

    return parser
if __name__ == "__main__":
    # Parse command line arguments
    parser = get_parser()
    args = parser.parse_args()

    # Define the base directory for JSON files
    json_path = "./json/"

    # Execute based on the provided command-line argument
    if args.model == 1:
        # Test the models and save results to a file
        test_res()
    elif args.model == 0:
        # Train the models and visualize the training progress
        loss1, loss2, loss3, acc1, acc2, acc3 = train_first(json_path+args.josn_name)
        losses1, losses2, losses3, losses4, acces1, acces2, acces3, acces4 = train_last(json_path + args.josn_name)
        losses = [loss1, loss2, loss3, losses1, losses2, losses3, losses4]
        accs = [acc1, acc2, acc3, acces1, acces2, acces3, acces4]
        draw_all(losses, accs)
    elif args.model == 2:
        # Load and visualize results from a JSON file
        res_show(json_path + args.josn_name)

