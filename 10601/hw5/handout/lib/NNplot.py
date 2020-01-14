import matplotlib.pyplot as plt


average_train_entropy = [0.583707762873, 0.244637574199, 0.175586084179, 0.164652851058, 0.173211636125]
average_test_entropy = [0.713573345913, 0.538644333782, 0.437204703457, 0.429913531649, 0.44056776685]


def plot(average_train_entropy, average_test_entropy):
    x = [5, 20, 50, 100, 200]
    a = plt.plot(x, average_train_entropy, linewidth = 3, color="red")
    b = plt.plot(x, average_test_entropy, linewidth = 3, color="blue")
    plt.xlabel('Number of hidden units')
    plt.ylabel('Average Cross Entropy')
    plt.title('Average_Cross_Entropy vs Number_of_hidden_units')
    plt.legend(('Train Average Cross Entropy', 'Test Average Cross Entropy '), loc='upper right')
    plt.show()
    return None

plot(average_train_entropy, average_test_entropy)