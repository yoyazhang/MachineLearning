import torch
from torch.autograd.variable import Variable
from torch.utils import data

from cnn import MyDataset


class Picture:
    def __init__(self, path, right_type):
        self.path = path
        self.type = right_type


def test(model, dataset):
    loader = data.DataLoader(dataset, shuffle=False)
    eval_acc = 0
    for k, (img, label) in enumerate(loader):
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        else:
            img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
            label = Variable(label)

        out = model(img)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()
    print('Accuracy: {:.6f}%'.format(eval_acc / len(dataset) * 100))


if __name__ == '__main__':
    m = torch.load("../models/part2_leakymore_withBatch.pth")
    final_test_set = []
    for i in range(1, 13):
        # home_path = "../train/" + str(i) + "/"
        home_path = "F:/test_data/" + str(i) + "/"
        # for j in range(1, 621):
        for j in range(1,241):
            path = home_path + str(j) + ".bmp"
            pic = Picture(path, i - 1)
            final_test_set.append(pic)
    final_data = MyDataset(final_test_set)
    test(m, final_data)
