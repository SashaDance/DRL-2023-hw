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

class RandomAgent():
    def __init__(self, action_n):
        self.action_n = action_n

    def get_action(self, state):
        action = np.random.randint(self.action_n)
        return action


class CrossEntropyAgent:
    def __init__(self, state_n, action_n):
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
                new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()

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

for q_param in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    for iteration_n in [150]:
        for trajectory_n in [1500]:

            '''
            this cycle is for eliminating too bad models:
            models with the same hyperparameters can have different
            mean_total_rewards because of the randomness
            '''
            mean_total_reward = 0

            print('----------------------------------------')
            print(q_param, iteration_n, trajectory_n)

            for i in range(3):
                print(i)
                agent = CrossEntropyAgent(state_n, action_n)

                for iteration in range(iteration_n):

                    # policy evaluation
                    trajectories = [get_trajectory(env, agent, max_len=200)
                                    for _ in range(trajectory_n)]
                    total_rewards = [np.sum(trajectory['rewards'])
                                     for trajectory in trajectories]

                    mean_total_reward = np.mean(total_rewards)

                    # policy improvement
                    quantile = np.quantile(total_rewards, q_param)
                    elite_trajectories = []
                    for trajectory in trajectories:
                        total_reward = np.sum(trajectory['rewards'])
                        if total_reward > quantile:
                            elite_trajectories.append(trajectory)

                    agent.fit(elite_trajectories)

                # checking if last model have ended the route properly
                if mean_total_reward >= 0:
                    break

            sum_ = 0
            n = 500

            for j in range(n):
                trajectory = get_trajectory(env, agent, max_len=200)
                sum_ += sum(trajectory['rewards'])

            sum_ = sum_ / n
            print(sum_)

            with open('hyper_params.txt', 'a') as f:
                f.write(f'{q_param} {iteration_n} {trajectory_n} {sum_}\n')

