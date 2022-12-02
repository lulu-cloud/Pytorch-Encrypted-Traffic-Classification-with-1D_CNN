from re import S
import torch.nn as nn
import torch
import torch.nn.functional as F
class OneCNN(nn.Module):
    def __init__(self,label_num):
        super(OneCNN,self).__init__()
        self.layer_1 = nn.Sequential(
            # 输入784*1
            nn.Conv2d(1,32,(1,25),1,padding='same'),
            nn.ReLU(),
            # 输出262*32
            nn.MaxPool2d((1, 3), 3, padding=0),
        )
        self.layer_2 = nn.Sequential(
            # 输入261*32
            nn.Conv2d(32,64,(1,25),1,padding='same'),
            nn.ReLU(),
            # 输入261*64
            nn.MaxPool2d((1, 3), 3, padding=0)
        )
        self.fc1=nn.Sequential(
            # 输入88*64
            nn.Flatten(),
            nn.Linear(87*64,1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024,label_num),
            nn.Dropout(p=0.3)
        )
    def forward(self,x):
        # print("x.shape:",x.shape)
        x=self.layer_1(x)
        # print("x.shape:",x.shape)
        x=self.layer_2(x)
        # print("x.shape:",x.shape)
        x=self.fc1(x)
        # print("x.shape:",x.shape)
        return x


class OneCNNC(nn.Module):
    def __init__(self,label_num):
        super(OneCNNC,self).__init__()
        self.layer_1 = nn.Sequential(
            # 输入784*1
            nn.Conv2d(1,32,(1,25),1,padding='same'),
            nn.ReLU(),
            # 输出262*32
            nn.MaxPool2d((1, 3), 3, padding=(0,1)),
        )
        self.layer_2 = nn.Sequential(
            # 输入262*32
            nn.Conv2d(32,64,(1,25),1,padding='same'),
            nn.ReLU(),
            # 输入262*64
            nn.MaxPool2d((1, 3), 3, padding=(0,1))
        )
        self.fc1=nn.Sequential(
            # 输入88*64
            nn.Flatten(),
            nn.Linear(88*64,1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024,label_num),
            nn.Dropout(p=0.3)
        )
    def forward(self,x):
        # print("x.shape:",x.shape)
        x=self.layer_1(x)
        # print("x.shape:",x.shape)
        x=self.layer_2(x)
        # print("x.shape:",x.shape)
        x=self.fc1(x)
        # print("x.shape:",x.shape)
        return x

class CNNImage(nn.Module):
    def __init__(self):
        super(CNNImage, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# x=torch.tensor([[1, 1,  0,  1,  2,  3],
#                 [1, 1,  4,  5,  6,  7],
#                 [1, 10, 8,  9, 10, 11]],dtype=torch.float32)
# x=x.reshape(1,3,-1)


# out_tensor=F.max_pool2d(x,(3,1),stride=3,padding=0)

# print(out_tensor)