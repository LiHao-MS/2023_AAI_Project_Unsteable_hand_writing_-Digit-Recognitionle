from dataset import HandwrittenDigitsDataset
from simply_model import HandwrittenDigitModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10000 # 根据需要调整epoch数量
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = HandwrittenDigitsDataset('../processed_data/train')  # 修改为你的train文件夹路径
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
model = HandwrittenDigitModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.NLLLoss()  # 使用负对数似然损失，因为我们使用了log_softmax作为输出层激活函数。如果使用softmax，请使用CrossEntropyLoss。
model.train()
losses = [] # 记录每个epoch的loss
SAVE_EVERY = 10  # 定义每隔多少个epoch保存一次模型

for epoch in range(NUM_EPOCHS):  # 训练循环...省略了部分代码，如验证等。请自行添加。
    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):  # 迭代加载数据
        data, target = data.to(DEVICE), target.to(DEVICE)  # 将数据转移到正确的设备上
        optimizer.zero_grad()  # 清空梯度缓存
        output = model(data)  # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 更新权重
        train_loss += loss.item()
        if batch_idx % 100 == 0:  # 每100个批次打印一次损失
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    losses.append(train_loss / len(train_loader))  # 计算并保存每个epoch的平均损失
    print('Epoch: {}, Average Train Loss: {:.6f}'.format(epoch, train_loss / len(train_loader)))
    if (epoch + 1) % SAVE_EVERY == 0:
        model_path = os.path.join('./models/sub_models/', f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}')

#python4.**保存模型**训练完成后，保存模型：
torch.save(model.state_dict(), 'models/handwritten_digit_model.pth')  # 修改为你想要的路径和文件名`5.**验证模型**加载验证数据并验证模型的正确率
plt.plot(losses)
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss.png')
val_dataset = HandwrittenDigitsDataset('../processed_data/val')# 修改为你的val文件夹路径
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
model.load_state_dict(torch.load('models/handwritten_digit_model.pth'))  # 加载模型
model.eval()  # 设置模型为评估模式
correct = 0
total = 0
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(val_loader):  # 迭代加载数据
        data, target = data.to(DEVICE), target.to(DEVICE)  # 将数据转移到正确的设备上
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)  # 获取预测值的索引，即预测的类别
        total += target.size(0)  # 累加总样本数
        correct += (predicted == target).sum().item()  # 计算并累加正确预测的样本数


print('Validation Accuracy: {:.4f}%'.format(100 * correct / total)) # 打印正确率