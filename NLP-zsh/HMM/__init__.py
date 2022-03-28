from HMMModel import Model

if __name__ == '__main__':
    hmm = Model(8)
    hmm.init_parameter()
    for i in range(0, 1000):
        hmm.train()
        print("ok train: " + str(i))
    hmm.print_parameters()
    hmm.generate_answers()
    for i in range(0, 11):
        print(hmm.answers[i])
