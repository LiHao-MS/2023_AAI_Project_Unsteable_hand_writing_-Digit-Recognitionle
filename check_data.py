import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data = np.load("./processed_data/processed_data/train/5/1110.npy")
print(data.shape)
# 遍历每张图像并显示  
for i in range(data.shape[0]):
    print(data[i])
    plt.figure(figsize=(1, 1))  # 设置图像大小  
    plt.imshow(data[i], cmap='gray')  # 使用灰度颜色映射  
    plt.title(f"Image {i+1}")  # 设置标题  
    plt.axis('off')  # 不显示坐标轴  
    plt.show()
plt.show()