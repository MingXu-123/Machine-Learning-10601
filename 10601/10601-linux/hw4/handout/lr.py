import sys
import time
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from feature import create_words_dict


def logistic_function(dot_product):
    return (1 / (1 + np.exp(-dot_product)))


def sparse_dot_product(example, theta_vec):
    product = 0.0
    for index in example:
        product += theta_vec[(index + 1)] * 1.0
    return (product + (theta_vec[0] * 1))[0]  # plus bias term


def parse_formatted_data(input_file):
    feature_lst = []
    labels = []
    with open(input_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            tmp_lst = []
            labels.append(int(row[0]))
            for feature_map in row[1:]:
                item = feature_map.split(":")
                tmp_lst.append(int(item[0]))
            feature_lst.append(tmp_lst)
    return feature_lst, labels


def get_Xi(example, length):
    xi = np.zeros([length, 1])
    for index in example:
        xi[index + 1] = 1
    xi[0] = 1  # bias term
    return xi


def average_log_likelihood(theta_vec, feature_lst_train, labels_train,
                           feature_lst_valid, labels_valid):
    res_train = 0
    N_train = len(labels_train)
    tmp_train = 0

    res_valid = 0
    N_valid = len(labels_valid)
    tmp_valid = 0

    for example, label in zip(feature_lst_train, labels_train):
        dot_product = sparse_dot_product(example, theta_vec)
        tmp_train = (-label * (dot_product)) + math.log(1 + np.exp(dot_product))
        res_train += tmp_train

    for example, label in zip(feature_lst_valid, labels_valid):
        dot_product = sparse_dot_product(example, theta_vec)
        tmp_valid = (-label * (dot_product)) + math.log(1 + np.exp(dot_product))
        res_valid += tmp_valid
    return ((1 / N_train) * (res_train)), ((1 / N_valid) * (res_valid))


def plot(average_log_likelihood_lst_train, average_log_likelihood_lst_valid):
    x = range(0, int(sys.argv[8]))
    a = plt.plot(x, average_log_likelihood_lst_train, color = "green")
    b = plt.plot(x, average_log_likelihood_lst_valid, color = "blue")
    plt.xlabel('Number of Epochs')
    plt.ylabel('Average Negative Log-Likelihood')
    plt.title('Average Negative LL vs Num_Of_Epochs')
    plt.legend(('Train Log Likelihood', 'Valid Log Likelihood'), loc='upper right')
    plt.show()
    return None


def train(theta_vec, feature_lst_train, labels_train, \
          feature_lst_valid, labels_valid, num_of_epoch):
    average_log_likelihood_lst_valid = []
    average_log_likelihood_lst_train = []
    epoch_lst = []

    for i in range(num_of_epoch):
        epoch_lst.append(i + 1)
    learning_rate = 0.1
    for epoch in range(num_of_epoch):
        for example, label in zip(feature_lst_train, labels_train):
            dot_product = sparse_dot_product(example, theta_vec)
            scalar = label - logistic_function(dot_product)
            Xi = get_Xi(example, len(theta_vec))
            theta_vec = np.add(theta_vec, (scalar * learning_rate) * Xi)
            # calculate average log-likelihood
        aver_log_likeli_train, aver_log_likeli_valid = average_log_likelihood(theta_vec, \
                                                                              feature_lst_train, labels_train,
                                                                              feature_lst_valid, labels_valid)
        average_log_likelihood_lst_train.append(aver_log_likeli_train)
        average_log_likelihood_lst_valid.append(aver_log_likeli_valid)
    return theta_vec, average_log_likelihood_lst_train, average_log_likelihood_lst_valid, epoch_lst


def error_predict(theta_vec, feature_lst, true_labels):
    num_of_error = 0
    predict_labels = []
    for i in range(len(feature_lst)):
        dot_product = sparse_dot_product(feature_lst[i], theta_vec)
        prob = logistic_function(dot_product)
        pred_label = 1 if prob >= 0.5 else 0
        predict_labels.append(pred_label)
        if pred_label != true_labels[i]:
            num_of_error += 1
    error_rate = num_of_error / len(true_labels)
    return predict_labels, error_rate


def write_predict_labels(predictions, output):
    res = ""
    for pred in predictions:
        res += str(pred) + "\n"
    res = res[:-1]
    with open(output, 'w') as f:
        f.write(res)


def write_metrics_out(metrics_out, trainError, testError):
    with open(metrics_out, 'w') as f:
        f.write('error(train): %.6f\n' % trainError)
        f.write('error(test): %.6f' % testError)


def main():
    train_input = sys.argv[1]
    vaild_input = sys.argv[2]
    test_input = sys.argv[3]
    dict_input = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    num_of_epoch = int(sys.argv[8])
    feature_lst_train, labels_train = parse_formatted_data(train_input)
    feature_lst_valid, labels_valid = parse_formatted_data(vaild_input)
    feature_lst_test, labels_test = parse_formatted_data(test_input)
    words_dict = create_words_dict(dict_input)
    lenth_of_theta_vec = len(words_dict)  # 39176
    theta_vec = np.zeros([lenth_of_theta_vec + 1, 1])  # 39177
    learned_theta_vector, average_log_likelihood_lst_train, average_log_likelihood_lst_valid, epoch_lst = \
        train(theta_vec, feature_lst_train, labels_train, \
              feature_lst_valid, labels_valid, num_of_epoch)
    plot(average_log_likelihood_lst_train, average_log_likelihood_lst_valid)
    predict_train_labels, train_error = error_predict(learned_theta_vector, \
                                                      feature_lst_train, labels_train)
    predict_test_labels, test_error = error_predict(learned_theta_vector, \
                                                    feature_lst_test, labels_test)
    write_metrics_out(metrics_out, train_error, test_error)
    write_predict_labels(predict_train_labels, train_out)
    write_predict_labels(predict_test_labels, test_out)


if __name__ == "__main__":
    main()



