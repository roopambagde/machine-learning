import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('AdsCTR_Optimisation.csv')
num_ads = data.shape[1] - 1
num_users = data.shape[0]
ads = data.values[:, :-1]
clicks = data.values[:, -1]
num_rewards_1 = np.zeros(num_ads)
num_rewards_0 = np.zeros(num_ads)
total_reward = 0
rewards = []
for i in range(num_users):
    ad = 0
    max_random = 0
    for j in range(num_ads):
        random_beta = np.random.beta(num_rewards_1[j] + 1, num_rewards_0[j] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = j
    reward = clicks[i]
    if reward == 1:
        num_rewards_1[ad] += 1
    else:
        num_rewards_0[ad] += 1
    total_reward += reward
    rewards.append(reward)
    if i % 1000 == 0:
        print("Iteration: ", i)
        print("Total Reward: ", total_reward)
        print("Num of 1s: ", num_rewards_1)
        print("Num of 0s: ", num_rewards_0)
        print("")
plt.plot(np.cumsum(rewards) / np.arange(1, num_users+1))
plt.xlabel('User')
plt.ylabel('CTR')
plt.show()

