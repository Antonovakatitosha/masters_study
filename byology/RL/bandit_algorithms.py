# banditsbook
from arms.bernoulli import BernoulliArm
import pandas as pd
from algorithms.epsilon_greedy.standard import EpsilonGreedy
from testing_framework.tests import test_algorithm
import matplotlib.pyplot as plt

# Define two adverts, with a probability of clicking from the users
# This is a simulation. Imagine that these are real ads.
arm0 = BernoulliArm(0.05)
arm1 = BernoulliArm(0.4)
arms = [arm0, arm1]

# print([arm1.draw() for _ in range(5)])

epsilon = 1  # Choose a random action every time
num_sims = 1000  # Number of repetitions
horizon = 250  # Length of experiment
# num_sims = 2  # Number of repetitions
# horizon = 3  # Length of experiment

df = pd.DataFrame()  # Buffer
for epsilon in [0, 0.1, 0.5, 1]:
        algo1 = EpsilonGreedy(epsilon, [], [])  # Algorithm

        sim_nums, times, chosen_arms, rewards, cumulative_rewards = test_algorithm(
            algo1, arms, num_sims, horizon)  # Running the environment/algorithm via the library
        # print(sim_nums)  # номер симуляции
        # print(times)  # времени (шаге)
        # print(chosen_arms)  # выбранном действии
        # print(rewards)  # полученной награде
        # print(cumulative_rewards)  # накопленной награде

        arrays = [[epsilon] * num_sims * horizon, sim_nums, times]  # Constructing the output array for aggregation
        # print(arrays)

        index = pd.MultiIndex.from_arrays(arrays, names=('epsilon', 'simulation', 'time'))
        # print(index)
        df_chosen_arm = pd.DataFrame(chosen_arms, index=index)
        # print(df_chosen_arm)

        # Это позволяет получить долю времени, в течение которой было выбрано каждое действие, относительно общего числа симуляций.
        df_chosen_arm = df_chosen_arm.groupby(level=[0, 2]).sum() / num_sims  # Aggregating to result in the proportion of time
        # print(df_chosen_arm)

        df = pd.concat([df, df_chosen_arm])  # Append to buffer.
        df.unstack(level=0).plot(ylim=[0, 1], ylabel="Probability of Optimal Action", xlabel="Steps")
print(df)
plt.show()
