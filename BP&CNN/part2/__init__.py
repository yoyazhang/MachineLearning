from torch.autograd.variable import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from cnn import CNNModel
import torch
from torch.utils import data
from cnn import MyDataset
import random
from test_in_class import test

TEST_SIZE = 155
learning_rate = 0.001


class Picture:
    def __init__(self, path, right_type):
        self.path = path
        self.type = right_type


if __name__ == '__main__':
    m1 = CNNModel()
    # 数据加载&分类
    test_set = []
    train_set = []
    for i in range(1, 13):
        home_path = "../train/" + str(i) + "/"
        # 从其中随机抽取30%(即186个)的数据用于测试，剩下70%(即434个)用于测试，分别放入两个dict中
        L = random.sample(range(0, 620), TEST_SIZE)
        for j in range(1, 621):
            path = home_path + str(j) + ".bmp"
            pic = Picture(path, i - 1)
            if j in L:
                test_set.append(pic)
            else:
                train_set.append(pic)
    train_data = MyDataset(train_set)
    test_data = MyDataset(test_set)
    train_loader = data.DataLoader(train_data, batch_size=25, shuffle=True)

    loss_func = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m1.parameters(), lr=learning_rate)
    # scheduler = ReduceLROnPlateau(opt, mode="min", patience=50, factor=0.9)

    loss_count = []
    for epoch in range(50):
        for i, (img, label) in enumerate(train_loader):
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()
            else:
                img = Variable(torch.unsqueeze(img, dim=1).float(), requires_grad=False)
                label = Variable(label)

            out = m1(img)
            # 获取损失
            loss = loss_func(out, label)
            print_loss = loss.data.item()
            # 使用优化器优化损失
            opt.zero_grad()  # 清空上一步残余更新参数值
            loss.backward()  # 误差反向传播，计算参数更新值
            opt.step()  # 将参数更新值施加到net的parameters上
        print("finish epoch: " + str(epoch))
        if epoch % 10 == 0:
            m1.eval()
            test(m1, test_data)

    m1.eval()
    test(m1, test_data)
    torch.save(m1, "../models/part2_leakymore_withBatch.pth")

    # relu_nobatch = 94.2%
    # relu-withbatch = 93%
    # sigmoid_withbatch = 97.15% 还有潜力，回去练久一点试试
    # lrdecay: 不知为何吊用没用，比乱猜还差
    # tanh-withbatch: 98.44%
    # leakyrelu: 99.1%
