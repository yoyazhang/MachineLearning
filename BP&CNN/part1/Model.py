import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import math


def sigmoid(num):
    return 1 / (1 + np.exp(-num))


def d_sigmoid(num):
    return num * (1 - num)


class Model:
    def __init__(self, node_nums, lr, data_size):
        if node_nums[0] != 1 or node_nums[len(node_nums) - 1] != 1:
            print("wrong format！")
            return
        self.r = lr
        self.level_num = len(node_nums)
        self.dataset = []
        for i in range(0, data_size):
            self.dataset.append(random.random() * math.pi * 2 - math.pi)
        self.result = []
        for i in range(len(self.dataset)):
            self.result.append(math.sin(self.dataset[i]))
        # 构建layer
        self.hidden_layers = []
        self.sigmoid_layers = []
        for i in range(0, self.level_num - 2):
            self.hidden_layers.append(Layer(node_nums[i + 1],node_nums[i],lr))
            # new
            self.sigmoid_layers.append(ActivationLayer())
            # old
            # if i != 0:
            #     self.sigmoid_layers.append(ActivationLayer())
        self.output_layer = Layer(node_nums[self.level_num - 1], node_nums[self.level_num - 2], lr)
        # 预测结果集
        self.predict_result = []

    def modify(self, index):
        delta_output = np.asarray(self.result[index] - self.predict_result[index])
        # print((delta_output.shape))
        delta_output = self.output_layer.alter(delta_output)
        # new
        j = len(self.sigmoid_layers) - 1
        delta_output = self.sigmoid_layers[j].alter(delta_output)
        i = len(self.hidden_layers) - 1
        j -= 1
        # old
        # j = len(self.sigmoid_layers) - 1
        while i >= 0:
            delta_output = self.hidden_layers[i].alter(delta_output)
            if i != 0:
                delta_output = self.sigmoid_layers[j].alter(delta_output)
                j -= 1
            i -= 1

    def conduct(self, num):
        ans = np.asarray(num)
        i = 0
        j = 0
        # print(len(self.hidden_layers))
        # print(len(self.sigmoid_layers))
        while i < len(self.hidden_layers):
            # print(i)
            ans = self.hidden_layers[i].conduct(ans)
            # new
            ans = self.sigmoid_layers[j].conduct(ans)
            j += 1
            i += 1
            # print("---")
            # print(ans)

        # print("k")
        ans = self.output_layer.conduct(ans)
        # print(ans)
        return ans[0][0]

    def train(self):
        total_loss = 0
        self.predict_result = []
        for i in range(0, len(self.dataset)):
            a = self.conduct(self.dataset[i])
            self.predict_result.append(a)
            # 计算准确率并打印本次训练记录让我有所了解
            loss = (self.predict_result[i] - self.result[i]) * (self.predict_result[i] - self.result[i])
            total_loss += loss
            # 修改模型
            self.modify(i)
        print("average loss: " + str(total_loss / len(self.dataset)))

    def draw(self):
        plt.plot(self.dataset, self.result, 'ro', color='red')
        # print(self.dataset)
        # print(self.predict_result)
        plt.plot(self.dataset, self.predict_result, 'ro', color='blue')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()


class Layer:
    def __init__(self, self_node_num, last_node_num, lr):
        self.node_num = self_node_num
        self.weight = random.randn(last_node_num, self_node_num)
        self.bias = random.randn(1, self_node_num)
        self.input = 1
        self.r = lr

    def conduct(self, input_x):
        # print("bbb")
        self.input = input_x
        return np.add(np.dot(input_x, self.weight), self.bias)

    def alter(self, delta_output):
        temp = np.dot(delta_output, self.weight.T)
        self.weight += self.r * np.dot(self.input.T, delta_output)
        self.bias += delta_output * self.r
        # 返回新的
        return temp


class ActivationLayer:
    def __init__(self):
        self.alter_parameter = []

    def conduct(self, input_x):
        temp = sigmoid(input_x)
        self.alter_parameter = d_sigmoid(temp)
        return sigmoid(input_x)

    def alter(self, delta_output):
        # sigmoid层实际上没有本身的参数调整，但需要将自己这层的导数与前面传递过来的结合并返回给后面
        # print(self.alter_parameter)
        return self.alter_parameter * delta_output
