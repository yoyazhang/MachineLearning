import torch
from CRFModel import Model

if __name__ == '__main__':
    # 用dataset1训练模型
    model = Model("train_dataset/dataset1/labels.utf8", "train_dataset/dataset1/train.utf8",
                  "train_dataset/dataset1/template.utf8", 1000)
    model.train()
    torch.save(model, "models/modelCRF-1.pth")

    # 用dataset2训练模型
    model = Model("train_dataset/dataset2/labels.utf8", "train_dataset/dataset2/train.utf8",
                  "train_dataset/dataset2/template.utf8", 1000)
    model.train()
    torch.save(model, "models/modelCRF-2.pth")

    # 训练bilstm模型
