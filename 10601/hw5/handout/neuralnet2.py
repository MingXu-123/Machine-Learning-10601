import numpy as np
from math import log
import sys
import matplotlib.pyplot as plt


#Class used to save parameters during back propagation
class parameters(object):
    def __init__(self, a, sigmoid_a, b, hat_y, J):
        self.a = a
        self.z = sigmoid_a
        self.b = b
        self.y_hat = hat_y
        self.J = J


def handleInput(inputFile):
    sample = []
    label = []
    with open(inputFile, 'r') as input:
        for line in input.readlines():
            sample_temp = [int(1)]  # bias term 1
            line = line.split(',')
            label_temp = np.zeros([1, 10])
            # print(label_temp)
            label_temp[0][int(line[0])] = 1
            label.append(label_temp)
            for i in line[1:]:
                sample_temp.append(int(i))
            sample.append(sample_temp)
        # print(np.array(sample))
        x_length = len(sample[0])
    return np.array(label), np.array(sample), x_length


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def getLength(inputFile):
    with open(inputFile, 'r') as input:
        return len(input.readlines())


def initParameters(classNumber, xLength, hiddenUnit, flag):
    if flag == 1:
        alpha = np.random.uniform(-0.1, 0.1, (hiddenUnit, xLength))
        beta = np.random.uniform(-0.1, 0.1, (classNumber, hiddenUnit + 1))
        alpha[:, 0] = 0.0
        beta[:, 0] = 0.0
        return alpha, beta

    elif flag == 2:
        alpha = np.zeros([hiddenUnit, xLength])
        beta = np.zeros([classNumber, hiddenUnit + 1])
        return alpha, beta


def softMaxForward(b):
    sum = 0
    temp = []
    for i in b:
        sum += np.exp(i)
    for ii in b:
        temp.append(np.exp(ii)/sum)
    return np.array(temp).T


def logHelper(hat_y):
    # log_hat_y = []
    # for line in hat_y:
    #     temp = []
    #     for element in line:
    #         temp.append(np.log(element))
    #     log_hat_y.append(temp)
    # return np.array(log_hat_y)
    [rows, cols] = hat_y.shape
    log_hat_y = []
    for hat_yi in hat_y.T:
        log_hat_y.append(log(hat_yi))
    new_log_hat_y = np.array([log_hat_y])
    return new_log_hat_y


def forwardComputation(sample_line, label_line, alpha, beta):
    a = np.dot(alpha, sample_line)
    a = np.insert(a, 0, 1) # add bias term
    a = np.array([a]).T
    [rows, cols] = a.shape
    # print([rows, cols])
    sigmoid_a = np.zeros((rows, cols))
    # print(rows, cols)
    for i in range(1, rows):
        for j in range(cols):
            sigmoid_a[i][j] = sigmoid(a[i][j])
    # sigmoid_a = np.array(sigmoid_a_temp)
    # print(sigmoid_a)
    sigmoid_a[0][0] = 1.0
    # print(sigmoid_a)
    b = np.dot(beta, sigmoid_a)
    hat_y = softMaxForward(b)
    log_hat_y = logHelper(hat_y)
    # print("log_hat_y is", log_hat_y)
    # print(label_line)
    J = -1 * np.dot(log_hat_y, label_line)
    # print("J is", J)
    return a, sigmoid_a, b, hat_y, J


def backComputation(sample, label, alpha, beta, obj): # Note that this beta is not the old one, it should not include bias term.
    # print(np.array([sample]))
    sample = np.array([sample])
    new_beta = np.delete(beta, 0, 1)  # remove first col
    # print(new_beta)
    new_z = np.delete(obj.z, 0, 0)  # remove first line
    # print(new_z)
    g_b = obj.y_hat.T - label  # 10*1
    # print(g_b)
    g_beta = np.dot(g_b, obj.z.T)
    # print('z')
    # print(obj.z.T)
    g_z = np.dot(new_beta.T, g_b)
    g_a = g_z * new_z * (1 - new_z)
    g_alpha = np.dot(g_a, sample)
    return g_alpha, g_beta


def calculateCrossEntropy(label, smaple, alpha, beta):
    count = 0
    temp = 0
    for j in zip(label, smaple):
        count += 1
        a, sigmoid_a, b, hat_y, J = forwardComputation(j[1], j[0].T, alpha, beta)
        temp += J
    return temp/count


def SGD(label_train, sample_train, label_test, sample_test, alpha, beta, learning_rate, epoch):
    # J_out_train = []
    # J_out_test = []
    J_out_train = 0
    J_out_test = 0
    for i in range(epoch):
        for j in zip(label_train, sample_train):
            a, sigmoid_a, b, hat_y, J = forwardComputation(j[1], j[0].T, alpha, beta)
            obj = parameters(a, sigmoid_a, b, hat_y, J)
            # print("J is ", obj.J)
            # print("y_hat is???????", obj.y_hat)
            g_alpha, g_beta = backComputation(j[1], j[0].T, alpha, beta, obj)
            alpha -= learning_rate * g_alpha
            beta -= learning_rate * g_beta

        J_mean_train = calculateCrossEntropy(label_train, sample_train, alpha, beta)
        J_mean_test = calculateCrossEntropy(label_test, sample_test, alpha, beta)
        J_out_train += J_mean_train
        J_out_test += J_mean_test
        # J_out_train.append(J_mean_train)
        # J_out_test.append(J_mean_test)

    return alpha, beta, J_out_train/100, J_out_test/100


def write_metrics(J_train, J_test, error_train, error_test):
    output_string = ""
    for i in range(len(J_test)):
        #print("train", len(J_train))
        #print("test", len(J_test))
        output_string += "epoch=" + str(i+1) + " " + "crossentropy(train): " + str(J_train[i][0][0]) + "\n"
        output_string += "epoch=" + str(i+1) + " " + "crossentropy(test): " + str(J_test[i][0][0]) + "\n"
    output_string += "error(train): " + str(error_train) + "\n"
    output_string += "error(test): " + str(error_test) + "\n"
    return output_string


def write_label(label):
    output_string = ""
    for i in label:
        output_string += str(i) + '\n'
    return output_string


def calculateError(label, sample, alpha, beta):
    label_predict = []
    count = 0
    error = 0
    for ii in zip(label, sample):
        a = np.dot(alpha, ii[1])
        a = np.insert(a, 0, 1)
        a = np.array([a]).T
        [rows, cols] = a.shape
        sigmoid_a = np.zeros((rows, cols))
        for i in range(1, rows):
            for j in range(cols):
                sigmoid_a[i][j] = sigmoid(a[i][j])
        sigmoid_a[0][0] = 1.0
        b = np.dot(beta, sigmoid_a)
        hat_y = softMaxForward(b)
        label_predict.append(np.where(hat_y == np.max(hat_y))[1][0])

        # print(np.where(hat_y == np.max(hat_y))[1][0])
        # print(label)
    for compare in zip(label_predict, label):
        count += 1
        if compare[0] != np.where(compare[1] == np.max(compare[1]))[1][0]:
            error += 1
    return label_predict, error/count


if __name__ == '__main__':
    classNumber = 10
    train_input = 'largeTrain.csv'
    test_input = 'largeValidation.csv'
    # train_out = sys.argv[3]
    # test_out = sys.argv[4]
    # metrics_out = sys.argv[5]
    num_epoch = 100
    hidden_units_list = [5] #, 20, 50, 100, 200]
    init_flag = 2
    learning_rate = 0.01
    # classNumber = 10
    # train_input = sys.argv[1]
    # test_input = sys.argv[2]
    # train_out = sys.argv[3]
    # test_out = sys.argv[4]
    # metrics_out = sys.argv[5]
    # num_epoch = int(sys.argv[6])
    # hidden_units = int(sys.argv[7])
    # init_flag = int(sys.argv[8])
    # learning_rate = float(sys.argv[9])
    label_train, sample_train, xLength = handleInput(train_input)
    label_test, sample_test, unused = handleInput(test_input)
    Y_train = []
    Y_test = []
    for hidden_units in hidden_units_list:
        alpha, beta = initParameters(classNumber, xLength, hidden_units, init_flag)
        result_alpha, result_beta, J_out_train, J_out_test = SGD(label_train, sample_train, label_test, sample_test, alpha, beta, learning_rate, num_epoch)
        # label_predict_train, error_train = calculateError(label_train, sample_train, result_alpha, result_beta)
        # label_predict_test, error_test = calculateError(label_test, sample_test, result_alpha, result_beta)
        Y_train.append(J_out_train[0][0])
        Y_test.append(J_out_test[0][0])


    # label_out_train = write_label(label_predict_train)
    # label_out_test = write_label(label_predict_test)
    # metrics_out_str = write_metrics(J_out_train, J_out_test, error_train, error_test)

    # print(metrics_out_str)
    # with open(train_out, 'w') as train:
    #     train.write(label_out_train)
    # with open(test_out, 'w') as test:
    #     test.write(label_out_test)
    # with open(metrics_out, 'w') as metrics:
    #     metrics.write(metrics_out_str)
    # print("train_error:", error)
    # print(result_alpha, result_beta)
    Y_train = np.array(Y_train)
    # print(Y_train)
    Y_test = np.array(Y_test)
    # print(Y_test)
    # X = np.array(hidden_units_list)
    # l1, = plt.plot(X, Y_train, color="blue", linewidth=2.5, linestyle="-", label='train')
    # l2, = plt.plot(X, Y_test, color="red", linewidth=2.5, linestyle="-", label='test')
    # plt.legend(loc='upper right')
    # plt.xlabel('# of Hidden Units')
    #
    # plt.show()
    plt.ylabel('Mean Cross Entropy')
    for i in Y_train:
        print("train",i)
    for j in Y_test:
        print("test",j)

