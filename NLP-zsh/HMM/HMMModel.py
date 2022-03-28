import numpy as np
import random


def list_to_array(X):
    # 将list(array)类型的数据展开成array类型
    c = []
    for i in range(len(X)):
        c += list(X[i])
    return np.array(c)


class Model:
    # status num算是一个超参
    def __init__(self, status_num):
        # 打开observation.utf8并按行读为数组
        file = open("observation.utf8")
        self.observations = []
        self.answers = []
        self.answer_probability = []
        for i in range(0, 10000):
            self.observations.append(file.readline()[0:7])
        file.close()
        # print(self.observations[0][0])
        # 状态转移矩阵,即为公式中的aij
        self.status_num = status_num
        self.A = np.zeros((status_num, status_num))
        # bij 表示在i状态下产出j的概率,一共10个数字(发射概率)
        self.prob_num = 10
        self.B = np.zeros((status_num, self.prob_num))
        # 初始状态矩阵(Pi)，表示初始状态最可能是什么
        self.PI = np.array([1.0 / status_num] * status_num)
        self.T = 7
        self.alpha = None
        self.beta = None
        self.Z = []
        self.observation_num = len(self.observations)
        for n in range(self.observation_num):
            self.Z.append(list(np.ones((len(self.observations[n]), self.status_num))))

    def init_parameter(self):
        for i in range(self.status_num):
            random_list = [random.randint(0, 100) for _ in range(self.status_num)]
            sum_of_random = sum(random_list)
            for j in range(self.status_num):
                self.A[i][j] = random_list[j] / sum_of_random
        for i in range(self.status_num):
            random_list = [random.randint(0, 100) for _ in range(10)]
            sum_of_random = sum(random_list)
            for j in range(10):
                self.B[i][j] = random_list[j] / sum_of_random

    # 估计序列X出现的概率
    def probability(self, observation):
        length = len(observation)
        Z = np.ones((length, self.status_num))
        # 向前向后传递因子
        c = self.forward(observation, Z)  # P(x,z)
        # 返回每一项概率之和
        return np.sum(np.log(c))

    def forward(self, observation, Z):
        # 计算前传矩阵，得到一个alpha
        # 前传矩阵的含义是：alpha[i][j]指在第i时刻的
        self.alpha = np.zeros((self.T, self.status_num))
        self.alpha[0] = self.emit_prob(observation[0]) * self.PI * Z[0]
        c = np.zeros(self.T)
        c[0] = np.sum(self.alpha[0])
        self.alpha[0] = self.alpha[0] / c[0]
        # 由前一排的值计算后面所有的值
        # 观察序列的第t位，因为已经知道
        for t in range(1, self.T):
            self.alpha[t] = self.emit_prob(observation[t]) * np.dot(self.alpha[t - 1], self.A) * Z[t]
            c[t] = np.sum(self.alpha[t])
            if c[t] == 0:
                continue
            self.alpha[t] = self.alpha[t] / c[t]
        return c

    def emit_prob(self, x):
        # 求x在状态k下的发射概率
        prob = np.zeros(self.status_num)
        for i in range(self.status_num):
            prob[i] = self.B[i][int(x[0])]
        return prob

    def backward(self, observation, Z, c):
        # 回传矩阵，得到一个beta
        self.beta = np.zeros((self.T, self.status_num))
        # 初始化
        self.beta[self.T - 1] = np.ones(self.status_num)
        for t in range(self.T - 2, -1, -1):
            self.beta[t] = np.dot(self.beta[t + 1] * self.emit_prob(observation[t + 1]), self.A.T) * Z[t]
            if c[t + 1] != 0:
                self.beta[t] = self.beta[t] / c[t + 1]

    def train(self):
        # 对于batch training，可以将其合并为一个
        total_a = np.zeros((self.status_num, self.status_num))
        total_pi = np.zeros(self.status_num)
        total_post_state = []
        # 初始化状态序列list

        for index in range(0, self.observation_num):
            # 对于每一个观察到的序列，调整status_initialize/status_change/emission
            c = self.forward(self.observations[index], self.Z[index])
            self.backward(self.observations[index], self.Z[index], c)
            post_state = self.alpha * self.beta / np.sum(self.alpha * self.beta)
            total_post_state.append(post_state)
            # 相邻状态的联合后验概率
            post_adj_state = np.zeros((self.status_num, self.status_num))
            for i in range(1, self.T):
                if c[i] == 0:
                    continue
                post_adj_state += (1 / c[i]) * np.outer(self.alpha[i - 1], self.beta[i] *
                                                        self.emit_prob(self.observations[index][i])) * self.A
            tmp_sum = np.sum(post_adj_state)
            if tmp_sum != 0:
                post_adj_state = post_adj_state / tmp_sum
            total_a += post_adj_state  # 批量累积：状态的后验概率
            # 累积初始概率,最后取均值
            total_pi += total_post_state[index][0]
        # 更新PI
        total_pi += 0.001 * np.ones(self.status_num)
        self.PI = total_pi / sum(total_pi)
        # 更新A，即状态转换概率
        total_a += 0.001
        for i in range(self.status_num):
            tmp_sum = np.sum(total_a[i])
            if tmp_sum != 0:
                self.A[i] = total_a[i] / tmp_sum
        # 更新B，即发射概率
        self.update_B(list_to_array(self.observations), list_to_array(total_post_state))

    def update_B(self, X, post_state):
        self.B = np.zeros((self.status_num, 10))
        length = len(X)
        for i in range(length):
            self.B[:, int(X[i])] += post_state[i]
        self.B += 0.01
        for k in range(self.status_num):
            tmp_sum = np.sum(post_state[:, k])
            if tmp_sum != 0:
                self.B[k] = self.B[k] / tmp_sum

    def print_parameters(self):
        print("Initial State Possibility")
        print(self.PI)
        print("Status change Possibility")
        print(self.A)
        print("Emission Possibility")
        print(self.B)

    def generate_answers(self):
        # 生成概率最高的10个答案
        for i in range(pow(10, 7)):
            if i % 10000 == 0:
                print(str(i))
            number = str(i)
            j = 7 - len(number)
            for m in range(0, j):
                number = "0" + number
            # 生成完毕，得到结果
            probability = self.probability(number)
            if len(self.answers) <= 10:
                self.answers.append(number)
                self.answer_probability.append(probability)
            else:
                min_index = -1
                min_prob = -1
                for k in range(1, 10):
                    if self.answer_probability[k] < probability:
                        if min_index == -1 or self.answer_probability[k] < min_prob:
                            min_index = k
                            min_prob = self.answer_probability[k]
                if min_index != -1:
                    self.answer_probability[min_index] = probability
                    self.answers[min_index] = number
