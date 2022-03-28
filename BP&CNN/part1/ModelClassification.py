import math

from Model import Layer
from Model import ActivationLayer
import random
import numpy as np
from PIL import Image

TEST_SIZE = 155


def softmax(score):
    m = np.max(score)
    score = np.exp(score - m)
    # score = np.exp(score)
    return score / np.sum(score)


def relu(x):
    return np.maximum(0, x)


def d_relu(x):
    return 0.01 if x < 0 else 1


def leaky_relu(x):
    return np.maximum(0.01 * x, x)


def d_leaky_relu(x):
    return 0.01 if x < 0 else 1


def tanh(x):
    exp = np.exp(x)
    return (exp - 1 / exp) / (exp + 1 / exp)


def d_tanh(x):
    tan = tanh(x)
    return 1 - tan ** 2


class ModelOfClassify:
    def __init__(self, node_nums, lr, lamda, lr_decay, step):
        if node_nums[0] != 784 or node_nums[len(node_nums) - 1] != 12:
            print("wrong format！")
            return
        self.r = lr
        self.step = step
        self.lr_decay = lr_decay
        self.lamda = lamda
        self.level_num = len(node_nums)
        # 构建layer
        self.hidden_layers = []
        self.activation_layers = []
        for i in range(0, self.level_num - 2):
            self.hidden_layers.append(SpecialLayer(node_nums[i + 1], node_nums[i], lr))
            if i % 2 == 1:
                self.activation_layers.append(LeakyReLuActivationLayer())
            else:
                self.activation_layers.append(ActivationLayer())
            # self.sigmoid_layers.append(TanhActivationLayer())
        self.output_layer = Layer(node_nums[self.level_num - 1], node_nums[self.level_num - 2], lr)
        self.test_set = []
        self.train_set = []
        self.final_test_set = []
        for i in range(1, 13):
            home_path = "../train/" + str(i) + "/"
            # 从其中随机抽取30%(即186个)的数据用于测试，剩下70%(即434个)用于测试，分别放入两个dict中
            L = random.sample(range(0, 620), TEST_SIZE)
            for j in range(1, 621):
                path = home_path + str(j) + ".bmp"
                pic = Picture(path, i - 1)
                if j in L:
                    self.test_set.append(pic)
                else:
                    self.train_set.append(pic)
        self.result_set = []
        for i in range(0, 12):
            tmp = []
            for j in range(0, 12):
                if j == i:
                    tmp.append(1)
                else:
                    tmp.append(0)
            self.result_set.append(np.asarray(tmp))
        # 面试用集
        for i in range(1, 13):
            home_path = "E:/test_data/" + str(i) + "/"
            for j in range(1, 241):
                path = home_path + str(j) + ".bmp"
                pic = Picture(path, i - 1)
                self.final_test_set.append(pic)

    def conduct(self, num):
        ans = np.asarray(num)
        i = 0
        while i < len(self.hidden_layers):
            ans = self.hidden_layers[i].conduct(ans)
            ans = self.activation_layers[i].conduct(ans)
            # print(ans)
            i += 1
        ans = softmax(self.output_layer.conduct(ans))
        return ans

    def modify(self, result, index):
        # 需要改
        delta_output = np.asarray(self.result_set[index] - result)
        # print(delta_output)
        delta_output = self.output_layer.alter(delta_output)
        # new
        i = len(self.hidden_layers) - 1
        while i >= 0:
            delta_output = self.activation_layers[i].alter(delta_output)
            delta_output = self.hidden_layers[i].alter(delta_output)
            i -= 1

    def train(self, time):
        # 一共12个汉字集需要训练
        random.shuffle(self.train_set)
        for i in range(0, len(self.train_set)):
            pic = self.train_set[i]
            # if i >= len(self.train_set) - 10:
            #     # print(pic.type)
            if time % self.step == 0:
                self.r *= self.lr_decay
            image = np.array(Image.open(pic.path).convert('L'), 'f')
            image *= 1 / 255
            # print(image)
            result = self.conduct(image.reshape([1, 784]))
            # print(result)
            # print("------")
            # 虽然标注的是1，但是实际上在result里对应的是第0个
            self.modify(result, pic.type)

    def test(self):
        print("------")
        total = 12 * TEST_SIZE
        right_num = 0

        for pic in self.test_set:
            # print(pic.path)
            image = np.array(Image.open(pic.path).convert('L'), 'f')
            image *= 1 / 255
            # print(image)
            result = self.conduct(image.flatten())[0]
            # print(str(result.tolist().index(max(result))) + "while right type:" + str(pic.type))
            # print("----")
            if result.tolist().index(max(result)) == pic.type:
                right_num += 1
        print("Accuracy: " + str(right_num / total * 100) + "%")

    def test_in_class(self):
        print("------")
        total = len(self.final_test_set)
        right_num = 0
        for pic in self.final_test_set:
            image = np.array(Image.open(pic.path).convert('L'), 'f')
            image *= 1 / 255
            result = self.conduct(image.flatten())[0]
            if result.tolist().index(max(result)) == pic.type:
                right_num += 1
        print("Accuracy: " + str(right_num / total * 100) + "%")


class Picture:
    def __init__(self, path, right_type):
        self.path = path
        self.type = right_type


class LeakyReLuActivationLayer(ActivationLayer):
    def conduct(self, input_x):
        # vfunc = np.vectorize(d_relu)
        temp = leaky_relu(input_x)
        # self.alter_parameter = d_relu(temp)
        self.alter_parameter = np.vectorize(d_leaky_relu)(temp)
        # print(self.alter_parameter)
        return temp


class ReLuActivationLayer(ActivationLayer):
    def conduct(self, input_x):
        # vfunc = np.vectorize(d_relu)
        temp = relu(input_x)
        # self.alter_parameter = d_relu(temp)
        self.alter_parameter = np.vectorize(d_relu)(temp)
        # print(self.alter_parameter)
        return temp


class TanhActivationLayer(ActivationLayer):
    def conduct(self, input_x):
        temp = tanh(input_x)
        self.alter_parameter = d_tanh(temp)
        return temp


class SpecialLayer(Layer):
    def __init__(self, self_node_num, last_node_num, lr):
        super().__init__(self_node_num, last_node_num, lr)
        self.weight *= 0.1
        self.bias *= 0.1
