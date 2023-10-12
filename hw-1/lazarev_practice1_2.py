import gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

env = gym.make('Taxi-v3')
state_n = 500
action_n = 6


class CrossEntropyAgent:
    def __init__(self, state_n, action_n, lamda=0.1,
                 policy_smoothing=True, alpha=0.2):
        self.policy_smoothing = policy_smoothing
        self.alpha = alpha
        self.lamda = lamda
        self.state_n = state_n
        self.action_n = action_n
        self.model = np.ones((self.state_n, self.action_n)) / self.action_n

    def get_action(self, state):
        action = np.random.choice(np.arange(self.action_n),
                                  p=self.model[state])
        return int(action)

    def fit(self, elite_trajectories):
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'],
                                     trajectory['actions']):
                new_model[state][action] += 1

        for state in range(self.state_n):
            if np.sum(new_model[state]) > 0:
                num_of_trajectories = np.sum(new_model[state])
                new_model[state] += self.lamda
                new_model[state] /= num_of_trajectories + self.lamda * action_n
            else:
                new_model[state] = self.model[state].copy()

        if self.policy_smoothing:
            self.model = (self.alpha * new_model +
                          (1 - self.alpha) * self.model)
        else:
            self.model = new_model

        return None


def get_trajectory(env, agent, max_len=1000, visualize=False):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    obs = env.reset()
    state = obs

    for _ in range(max_len):
        trajectory['states'].append(state)

        action = agent.get_action(state)
        trajectory['actions'].append(action)

        obs, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)

        state = obs

        if visualize:
            time.sleep(0.5)
            env.render()

        if done:
            break

    return trajectory


agent = CrossEntropyAgent(state_n, action_n)

mean_total_reward = 0

q_param = 0.4
iteration_n = 500
trajectory_n = 1500


for iteration in range(iteration_n):

    # policy evaluation
    trajectories = [get_trajectory(env, agent, max_len=200)
                    for _ in range(trajectory_n)]
    total_rewards = [np.sum(trajectory['rewards'])
                     for trajectory in trajectories]

    mean_total_reward = np.mean(total_rewards)

    print(f'iteration: {iteration}, reward: {mean_total_reward}')
    # policy improvement
    quantile = np.quantile(total_rewards, q_param)
    elite_trajectories = []
    for trajectory in trajectories:
        total_reward = np.sum(trajectory['rewards'])
        if total_reward > quantile:
            elite_trajectories.append(trajectory)

    agent.fit(elite_trajectories)

sum_ = 0
n = 500

for j in range(n):
    trajectory = get_trajectory(env, agent, max_len=200, visualize=True)
    sum_ += sum(trajectory['rewards'])

print(sum_ / n)
