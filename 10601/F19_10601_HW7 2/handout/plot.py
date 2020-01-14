import matplotlib.pyplot as plt




train_accuracy = [0.83281685, 0.83368611, 0.86076583, 0.93786495]
test_accuracy = [0.83250263, 0.83355399, 0.85646976, 0.92256921]




def plot(train_accuracy, test_accuracy):
    x = [10, 100, 1000, 10000]
    a = plt.plot(x, train_accuracy, color="green")
    b = plt.plot(x, test_accuracy, color="blue")
    plt.xlabel('Number of sequences')
    plt.ylabel('Accuracy')
    plt.title('train_accuracy_and_test_accuracy')
    plt.legend(('train_accuracy', 'test_accuracy'), loc='upper')
    plt.show()
    return None

plot(train_accuracy, test_accuracy)