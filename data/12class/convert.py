import matplotlib.pyplot as plt
import gzip
import numpy as np
import os

def load_data_gz(data_folder):
    files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz']

    paths = []
    for fname in files:
        paths.append(os.path.join(data_folder, fname))

    # 读取每个文件夹的数据
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 784)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 784)

    return x_train, y_train, x_test, y_test

# 调用load_data_gz函数加载数据集
data_folder = r'data/12class/SessionAllLayers'
x_train_gz, y_train_gz, x_test_gz, y_test_gz = load_data_gz(data_folder)

print('x_train_gz.shape:', x_train_gz.shape)
print('y_train_gz.shape', y_train_gz.shape)
print('x_test_gz.shape:', x_test_gz.shape)
print('y_test_gz.shape:', y_test_gz.shape)

# 784->28*28
train_image = np.zeros([x_train_gz.shape[0], 28, 28]).astype(np.float32)

for i in range(x_train_gz.shape[0]):
    re = x_train_gz[i, :].reshape(28, 28)
    train_image[i, :, :] = re
print('train_image.shape: ', train_image.shape)

# 选择前n张进行查看
n=20
plt.figure()
for i in range(n):
    plt.subplot(5, 4, i+1)
    plt.imshow(train_image[i, :, :], 'gray')
    plt.axis('off')
plt.show()

