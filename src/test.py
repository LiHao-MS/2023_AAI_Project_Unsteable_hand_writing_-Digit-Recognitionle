import torch
import matplotlib.pyplot as plt
import numpy as np

# 创建一个从-5到5的等差数列作为输入
x = torch.linspace(-5, 5, 1000).unsqueeze(1)  # unsqueeze用于增加一维，使其适应logsoftmax函数的输入格式

# 应用logsoftmax函数
y = torch.log_softmax(x, dim=1)
y1 = torch.softmax(x, dim=1)
# 将torch.tensor转换为numpy数组以便于matplotlib绘图
x_numpy = x.squeeze().numpy()
y_numpy = y.squeeze().numpy()
y1_numpy = y1.squeeze().numpy()
# 绘制图像
plt.figure(figsize=(8, 6))
plt.plot(x_numpy, y_numpy, label='logsoftmax')
plt.plot(x_numpy, y1_numpy, label='softmax')
plt.title('LogSoftmax Function Applied to a Single Dimension')
plt.xlabel('Input Values')
plt.ylabel('Log Softmax Output')
plt.grid(True)
plt.show()