import numpy as np
from PIL import Image
import torch.utils.data as data
import torch
from torch import nn

TEST_SIZE = 155


class MyDataset(data.Dataset):
    def __init__(self, pictures):
        self.pictures = pictures

    def __getitem__(self, index):
        image = np.array(Image.open(self.pictures[index].path).convert('L'), 'f')
        image *= 1 / 255

        label = self.pictures[index].type
        return image, label

    def __len__(self):
        return len(self.pictures)


class CNNModel(torch.nn.Module):
    def __init__(self):
        # 制作数据集，这和1类似
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=3),
            nn.BatchNorm2d(25),
            nn.LeakyReLU()
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=3),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(50 * 5 * 5, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 12)
        )

    def forward(self, x):
        # print(x.shape)
        x = self.layer1(x)

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
