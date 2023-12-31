from dataset import HandwrittenDigitsDataset
from CNN_model import CnnM
from mlp_model import MLP
import torch
from torch.utils.data import DataLoader, TensorDataset
from test_dataset import TestDataset
from utils import IdentityTransform

# Define hyperparameters for test
BATCH_SIZE = 100  # 根据需要调整batch大小
LEARNING_RATE = 0.001  # 学习率
NUM_EPOCHS = 60 # 根据需要调整epoch数量
DRO_NUM_EPOCH = 400
WEIGHT_DECAY = 0.001  # 学习率衰减
HIDDEN_DIM = 300

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1 = CnnM(input_channel=10, hidden_dim=HIDDEN_DIM).to(DEVICE)
model2 = MLP(input_dim=HIDDEN_DIM, output_dim=10).to(DEVICE)

def test_acc(model1, model2,model_path, loader):
    """
        Function to calculate accuracy on a given data loader using provided models.

        Args:
            model1 (torch.nn.Module): The first part of the model pipeline.
            model2 (torch.nn.Module): The second part of the model pipeline.
            model_path (str): Path to the directory containing saved model states.
            loader (torch.utils.data.DataLoader): Data loader for the dataset to evaluate on.

        Returns:
            float: Accuracy value.
    """
    model1.load_state_dict(torch.load(model_path+'1.pth', map_location=DEVICE))
    model2.load_state_dict(torch.load(model_path+'2.pth', map_location=DEVICE))
    model1.eval()
    model2.eval()
    correct = 0
    total = 0

    # Compute accuracy with no gradients computed
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):  # 迭代加载数据
            data, target = data.to(DEVICE), target.to(DEVICE)  # 将数据转移到正确的设备上
            outputs = model2(model1(data))
            correct += torch.sum((torch.argmax(outputs, dim=1) == target).float()).item()
            total += target.size(0)  # 累加总样本数
    return correct / total

def get_acc_loss():
    """
    Function to compute train and validation accuracy for multiple models.

    Returns:
        None: Prints accuracy values for each model.
    """
    # Create train and validation datasets and loaders
    train_dataset = HandwrittenDigitsDataset('../processed_data/train', transform=IdentityTransform())
    val_dataset = HandwrittenDigitsDataset("../processed_data/val", transform=IdentityTransform())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Define paths to different model checkpoints
    fake_model_path = './models/type_fake_final_model'
    type3v1_model_path = './models/type3v1_final_model'
    type3v2_model_path = './models/type3v2_final_model'
    type4v1_model_path = './models/type4v1_final_model'
    type4v2_model_path = './models/type4v2_final_model'

    # Calculate and print train accuracies
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

    # Calculate and print validation accuracies
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
    """
       Function to test the final model on the test dataset and write predictions to a file.

       Returns:
           None: Writes predictions to "../res.txt" file.
    """
    # Define path to the final model checkpoint
    model_path = './models/type4v2_final_model'
    test_dataset = TestDataset('../processed_data/test')
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE, shuffle=False)
    model1.load_state_dict(torch.load(model_path+'1.pth', map_location=DEVICE))
    model2.load_state_dict(torch.load(model_path+'2.pth', map_location=DEVICE))
    model1.eval()
    model2.eval()
    # Write predictions to a file
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

    return
if __name__ == "__main__":
    # Run the function to test the final model on the test dataset and save results
    test_res()
