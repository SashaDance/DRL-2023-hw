import gym
import numpy as np
import time

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
                new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()

        self.model = new_model
        return None


def get_deterministic_policy(agent: CrossEntropyAgent) -> np.ndarray:

    policy = agent.model

    deterministic_policy = np.zeros((state_n, action_n))
    for state in range(state_n):
        action = np.random.choice(np.arange(action_n),
                                  p=policy[state])
        deterministic_policy[state][action] = 1

    return deterministic_policy


def get_det_trajectory(deterministic_policy: np.ndarray, max_len=200,
                       visualize=False) -> dict:
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    obs = env.reset()
    state = obs

    for _ in range(max_len):
        trajectory['states'].append(state)

        action = np.random.choice(np.arange(action_n),
                                  p=deterministic_policy[state])

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


def get_elite_trajectories(agent: CrossEntropyAgent,
                           q_param: float = 0.2,
                           m: int = 100, k: int = 100) -> list[dict]:

    elite_trajectories = []
    trajectories = []
    evaluations = []
    total_rewards = []

    for i in range(m):
        policy = get_deterministic_policy(agent)
        fixed_m_trajectories = []
        sum_ = 0
        for j in range(k):
            trajectory = get_det_trajectory(policy)
            fixed_m_trajectories.append(trajectory)
            sum_ += np.sum(trajectory['rewards'])

        evaluation = sum_ / k
        evaluations.append(evaluation)
        for trajectory in fixed_m_trajectories:
            trajectory['evaluation'] = evaluation

        trajectories = trajectories + fixed_m_trajectories

    quantile = np.quantile(evaluations, q_param)

    for trajectory in trajectories:
        if trajectory['evaluation'] > quantile:
            elite_trajectories.append(trajectory)

        total_rewards.append(np.sum(trajectory['rewards']))

    mean_total_reward = np.mean(total_rewards)
    print(f'Mean total reward: {mean_total_reward}')

    return elite_trajectories


def get_trajectory(agent, max_len=200, visualize=False):
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
pol = get_deterministic_policy(agent)

q_param = 0.6
iteration_n = 200
m = 100
k = 5

for iteration in range(iteration_n):

    print(f'Iteration: {iteration}')
    elite_trajectories = get_elite_trajectories(agent, q_param, m, k)
    agent.fit(elite_trajectories)

sum_ = 0
n = 500

for j in range(n):
    trajectory = get_trajectory(agent, max_len=200)
    sum_ += sum(trajectory['rewards'])

print(sum_ / n)
