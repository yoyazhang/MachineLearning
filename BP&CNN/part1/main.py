from Model import Model
from ModelClassification import ModelOfClassify, Picture
import matplotlib.image as img
from ModelClassification import d_relu
import torch as pt
import numpy as np

from PIL import Image

if __name__ == '__main__':
    # m1 = Model([1, 50, 1], 0.005, 1200)
    # for i in range(0, 2000):
    #     m1.train()
    #     print("---")
    # m1.draw()
    #
    # m2 = Model([1, 10, 15, 1], 0.005, 1200)
    # for i in range(0, 2000):
    #     m2.train()
    #     print("---")
    # m2.draw()
    #
    # m3 = Model([1, 10, 15, 10, 1], 0.005, 1200)
    # for i in range(0, 2000):
    #     m3.train()
    #     print("---")
    # m3.draw()

    # train
    # m2 = ModelOfClassify([784, 512, 256, 128,64, 12], 0.01, 0, 0.99, 20)
    # m2.r = 0.001

    # 检查一下
    # m2 = pt.load("../models/model_test_3-simple-relu.pth")# 73%
    # m2 = pt.load("../models/model_test_4-leakyrelu.pth")# 82%
    # m2 = pt.load("../models/model_test_4-tanh&relu.pth")# 73%
    # m2 = pt.load("../models/model_test-4-sigmoid&leakyrelu-256-2.pth")# 89%
    # m2 = pt.load("../models/model_test-5-sigmoid&leakyrelu-256-128.pth") # 91.6%
    m2 = pt.load("../models/model_test-5-sigmoid&leakyrelu-512-256-128.pth")
    final_test_set = []
    for i in range(1, 13):
        home_path = "F:/test_data/" + str(i) + "/"
        for j in range(1, 241):
            path = home_path + str(j) + ".bmp"
            pic = Picture(path, i - 1)
            final_test_set.append(pic)

    print("------")
    total = len(final_test_set)
    right_num = 0
    for pic in final_test_set:
        image = np.array(Image.open(pic.path).convert('L'), 'f')
        image *= 1 / 255
        result = m2.conduct(image.flatten())[0]
        if result.tolist().index(max(result)) == pic.type:
            right_num += 1
    print("Accuracy: " + str(right_num / total * 100) + "%")
    # for i in range(0, 150):
    #     m2.train(i)
    #     m2.test()
    # m2.test_in_class()
    # print('final result:')
    # pt.save(m2, "../models/model_test-6-sigmoid&leakyrelu-512-256-128.pth")
