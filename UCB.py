import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Load the data from CSV
data = pd.read_csv('AdsCTR_Optimisation.csv')
num_ads = data.shape[1] - 1
num_users = data.shape[0]
ads = data.values[:, :-1]
clicks = data.values[:, -1]

# Define the agent
counts = np.zeros(num_ads) # Number of times each ad has been selected
sum_rewards = np.zeros(num_ads) # Sum of rewards for each ad
total_reward = 0
ucb_values = np.zeros(num_ads) # UCB values for each ad
for i in range(num_ads):
    ad = i
    reward = clicks[ad]
    counts[ad] += 1
    sum_rewards[ad] += reward
    total_reward += reward

# Define the UCB loop
rewards = []
for i in range(1, num_users):
    ad = 0
    max_ucb = 0
    for j in range(num_ads):
        if counts[j] > 0:
            average_reward = sum_rewards[j] / counts[j]
            delta = np.sqrt(2 * np.log(i) / counts[j])
            ucb = average_reward + delta
        else:
            ucb = 1e400
        ucb_values[j] = ucb
        if ucb > max_ucb:
            max_ucb = ucb
            ad = j
    reward = clicks[ad]
    counts[ad] += 1
    sum_rewards[ad] += reward
    total_reward += reward
    rewards.append(reward)

# Plot the rewards over time
plt.plot(np.cumsum(rewards) / np.arange(1, num_users))
plt.xlabel('User')
plt.ylabel('CTR')
plt.show()
