import numpy as np
import matplotlib.pyplot as plt

# 读取npz文件
# data = np.load('../processed_data/val/0/25.npy')
#
# # 假设你的npz文件中有一个名为'tensor'的键对应10x28x28的tensor
# # tensor = data['tensor']
#
# fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(10, 10))
# for i in range(10):
#     img = data[i].reshape(28, 28)
#     axes[i].imshow(img, cmap='gray')
#     axes[i].axis('off')
#
# plt.show()
d1 = [str(i*2) for i in range(5)]
d2 = [str(i*2+1) for i in range(5)]
for i, j in zip(d1, d2):
    print(i, j)