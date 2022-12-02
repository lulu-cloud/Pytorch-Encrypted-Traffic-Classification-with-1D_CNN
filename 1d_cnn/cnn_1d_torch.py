from random import shuffle
import time
import sys
import torch.nn as nn
import numpy as np
import os

import torchvision

from model import OneCNN,CNNImage,OneCNNC
from torchvision import datasets,transforms
import gzip
import torch
from data import DealDataset


def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 设置超参数
    batch_size = 50
    lr = 1.0e-4
    num_epochs = 40
    # label_num = 12
    label_num=12


    # 导入数据
    folder_path_list=[
        r"data/12class/FlowAllLayerss",
        r"data/12class/FlowL7",
        r"data/12class/SessionAllLayers",
        r"data/12class/SessionL7",
                      ]

    # task_index 可以取 0，1，2，3
    task_index = 0

    folder_path = folder_path_list[task_index]
    train_data_path = "train-images-idx3-ubyte.tgz"
    train_label_path = "train-labels-idx1-ubyte.tgz"
    test_data_path = "t10k-images-idx3-ubyte.tgz"
    test_label_path = "t10k-labels-idx1-ubyte.tgz"

    trainDataset = DealDataset(folder_path,train_data_path,train_label_path)
    testDataset = DealDataset(folder_path,test_data_path,test_label_path)

    train_loader = torch.utils.data.DataLoader(
        dataset=trainDataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=testDataset,
        batch_size=batch_size,
        shuffle=False
    )

    # 定义模型
    model = OneCNNC(label_num)
    model = model.to(device)
    # model = CNNImage()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # images=images.reshape(-1,1,28,28)
            images = images.to(device)
            labels = labels.to(device)
            # print(images.shape)
            # print(labels.shape)
            # Forward pass
            outputs = model(images.to(torch.float32))
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        test_length = len(testDataset)
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images.to(torch.float32))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model on the {} test images: {} %'.format(test_length,100 * correct / total)) 

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

if __name__=='__main__':
    main()

