from environment import MountainCar
import sys
import random
import numpy as np
import matplotlib.pyplot as plt


def linear_approximation(state, weights, bias):
    Q_value = [] # Q_value list represents Q value for action 0, action 1, action2
    (row_of_weights, col_of_weights) = weights.shape
    for col in range(col_of_weights):
        tmp_q_value = float(0)
        for row in state:
            tmp_q_value += weights[row][col] * state[row]
        update_q_value = tmp_q_value + (bias * 1) #add bias term
        Q_value += [update_q_value]
    return Q_value


def greedy_action_selection(Q_value, epsilon):
    prob = np.random.uniform(0, 1)
    if prob <= epsilon:
        random_action = random.choice([0, 1, 2])
        return random_action
    else:
        max_q = max(Q_value)
        max_action = Q_value.index(max_q)
        return max_action


def compute_TD_target(s_next, weights, bias, reward, gamma):
    Q_value_next = linear_approximation(s_next, weights, bias)
    TD_target = reward + (gamma * max(Q_value_next))
    return TD_target


def compute_gradiant(action, weights, state):
    (row_of_weights, col_of_weights) = weights.shape
    gradiant_m = np.zeros([row_of_weights, col_of_weights])
    # the value of ath col is the value of state s
    for key in state:
        gradiant_m[key][action] = state[key]
    return gradiant_m


def train(num_episodes, max_iterations, car, weights, bias, epsilon, gamma, learning_rate):
    rewards_sum_lst = []
    for episode in range(num_episodes):
        state = car.reset()
        rewards_sum = float(0)
        for iteration in range(max_iterations):
            Q_value = linear_approximation(state, weights, bias)
            action = greedy_action_selection(Q_value, epsilon)
            (s_next, reward, done) = car.step(action)
            rewards_sum += reward
            TD_target = compute_TD_target(s_next, weights, bias, reward, gamma)
            gradiant_m = compute_gradiant(action, weights, state)
            weights = weights - learning_rate * (Q_value[action] - TD_target) * gradiant_m
            g_bias = 1
            bias = bias - learning_rate * (Q_value[action] - TD_target) * g_bias
            state = s_next
            if (done == True): break
        rewards_sum_lst.append(rewards_sum)
    return weights, bias, rewards_sum_lst


def write(out_path, str_output):
    with open(out_path, 'w') as f:
        f.write(str_output)



def plot(rewards_sum_lst, rolling_mean):
        x = [i for i in range(2000)]
        x1 = [i for i in range(25, 2025, 25)]
        a = plt.plot(x, rewards_sum_lst, color="green")
        b = plt.plot(x1, rolling_mean, color="blue")
        plt.xlabel('Number of episodes')
        plt.ylabel('rewards')
        plt.title('raw_Return_and_rolling_mean')
        plt.legend(('Return', 'Rolling_mean'), loc='upper')
        plt.show()
        return None


def main(args):
    mode = args[1]
    weight_out = args[2]
    returns_out = args[3]
    num_episodes = int(args[4])
    max_iterations = int(args[5])
    epsilon = float(args[6])
    gamma = float(args[7])
    learning_rate = float(args[8])

    num_row = MountainCar(mode).state_space
    num_col = MountainCar(mode).action_space
    car = MountainCar(mode)
    bias = 0
    weights = np.zeros([num_row, num_col])
    update_weight, bias, rewards_sum_lst = train(num_episodes, max_iterations, car,\
                                                 weights, bias, epsilon, gamma, learning_rate)
    returns_out_str = ""
    for item in rewards_sum_lst:
        returns_out_str += str(item) + "\n"
    returns_out_str = returns_out_str[:-1]
    write(returns_out, returns_out_str)
    weights_str = ""
    weights_str += str(bias) + "\n"
    (row, col) = update_weight.shape
    for i in range(row):
        for j in range(col):
            weights_str += str(float(update_weight[i][j])) + "\n"
    weights_str = weights_str[:-1]
    write(weight_out, weights_str)
    rolling_mean = []
    count = 0
    reward_mean = 0
    for i in range(len(rewards_sum_lst)):
        if (count == 25):
            rolling_mean.append(reward_mean/25)
            reward_mean = 0
            count = 1
            reward_mean += rewards_sum_lst[i]
            continue
        reward_mean += rewards_sum_lst[i]
        count += 1

    res = rewards_sum_lst[1975:]
    print(len(res))

    rolling_mean.append(sum(res) / 25)
    print(len(rolling_mean))
    plot(rewards_sum_lst, rolling_mean)




if __name__ == "__main__":
    main(sys.argv)



