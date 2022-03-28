class Feature:
    def __init__(self, labels, template):
        self.weights = {}
        self.labels = labels
        self.template = template

    def compare(self, input_key, result):
        for i in range(len(self.weights)):
            result[i] += self.weights[self.labels[i]][input_key]
        return result

    def modify(self, good_label, bad_label, input_key):
        self.weights[good_label][input_key] += 1
        self.weights[bad_label][input_key] -= 1


class UFeature(Feature):
    def __init__(self, labels, template, words):
        super().__init__(labels, template)
        for i in range(len(labels)):
            hashset = {}
            for j in range(len(words)):
                key = ""
                for tmp in template:
                    index = j + tmp
                    if index < 0:
                        key += "NIL"
                    else:
                        key += words[index]
                hashset[key] = 1
            self.weights[labels[i]] = hashset


class BFeature(Feature):
    def __init__(self, labels, template, words):
        super().__init__(labels, template)
        self.labels.append("NIL")
        for i in range(len(labels)):
            hashset = {}
            for k in range(len(labels)):
                last_word_label = labels[k]
                for j in range(len(words)):
                    key = ""
                    for tmp in template:
                        index = j + tmp
                        if index < 0:
                            key += "NIL"
                        else:
                            key += words[index]
                    hashset[(key, last_word_label)] = 1
            self.weights[labels[i]] = hashset


def right_prob(answer, right_answer):
    right_num = 0
    if answer is not None:
        for i in range(len(answer)):
            if answer[i] == right_answer[i]:
                right_num += 1
        print(f"准确率：{right_num / len(right_answer)}")
    else:
        print("暂无结果")


class Model:
    def __init__(self, label_path, train_set_path, template_path, epoch):
        self.epoch = epoch
        # 加载label集
        self.labels = []
        file = open(label_path)
        while 1:
            line = file.readline()
            if line is None:
                break
            self.labels.append(line[0])
        # 加载训练集
        self.dataset = []
        self.dataset_answer = []
        # 生成dataset及其answer用于训练
        file = open(train_set_path)
        while 1:
            line = file.readline()
            if line is None:
                break
            self.dataset.append(line[0])
            self.dataset_answer.append(line[2])
        file.close()
        # 生成特征函数
        self.u_templates = []
        self.b_templates = []
        self.readTemplates(template_path)
        self.u_features = []
        self.b_features = []
        self.generate_u_features()
        self.generate_b_features()

    def readTemplates(self, template_path):
        file = open(template_path, encoding='utf-8')
        isUniGram = True
        for readLine in file:
            tmp_list = []
            if readLine.find("Unigram") > 0 or readLine.find("Bigram") > 0:
                continue
            if isUniGram:
                if len(readLine.strip()) == 0:
                    isUniGram = False
                else:
                    if readLine.find("/") > 0:
                        left = readLine.split("/")[0].split("[")[-1].split(",")[0]
                        right = readLine.split("/")[-1].split("[")[-1].split(",")[0]
                        tmp_list.append(int(left))
                        tmp_list.append(int(right))
                    else:
                        num = readLine.split("[")[-1].split(",")[0]
                        tmp_list.append(int(num))
                    self.u_templates.append(tmp_list)
            else:
                if readLine.find("/") > 0:
                    left = readLine.split("/")[0].split("[")[-1].split(",")[0]
                    right = readLine.split("/")[-1].split("[")[-1].split(",")[0]
                    tmp_list.append(int(left))
                    tmp_list.append(int(right))
                else:
                    num = readLine.split("[")[-1].split(",")[0]
                    tmp_list.append(int(num))
                self.b_templates.append(tmp_list)

    def generate_u_features(self):
        for template in self.u_templates:
            self.u_features.append(UFeature(self.labels,template,self.dataset))

    def generate_b_features(self):
        for template in self.b_templates:
            self.b_features.append(BFeature(self.labels,template,self.dataset))

    def train(self):
        for i in range(self.epoch):
            answer = self.conduct(self.dataset)
            self.modify(answer, self.dataset_answer)

    def modify(self, answer, right_answer):
        for i in range(len(answer)):
            if answer[i] == right_answer[i]:
                continue
            else:
                if i == 0:
                    last_type = "NIL"
                else:
                    last_type = answer[i-1]
                for feature in self.u_features:
                    # 考虑模板需要什么词，传给特征对象
                    feature.modify(self.generate_key_u(i, feature.template), right_answer[i], answer[i])
                for feature in self.b_features:
                    feature.modify((self.generate_key_u(i, feature.template), last_type), right_answer[i], answer[i])

    def generate_key_u(self, j, template):
        key = ""
        for tmp in template:
            index = j + tmp
            if index < 0:
                key += "NIL"
            else:
                key += self.dataset[index]
        return key

    def conduct(self, source):
        # 在训练阶段，source就是dataset
        answer = []
        for i in range(len(source)):
            score = [0 for _ in range(len(self.labels))]
            if i == 0:
                last_type = "NIL"
            else:
                last_type = answer[i-1]
            for feature in self.u_features:
                score = feature.compare(self.generate_key_u(i, feature.template), score)
            for feature in self.b_features:
                score = feature.compare((self.generate_key_u(i, feature.template), last_type), score)
            answer.append(self.labels[score.index(max(score))])
        return answer
