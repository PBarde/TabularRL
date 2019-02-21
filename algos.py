import numpy as np


# todo : ask questions
# 1 - what is the value of gamma ?
# 2 - do we evaluate always from same initial state ?
# 3 - maybe be more specific on the plots we are suppose to do ("see the book" what page?)
# 4 - what do you mean by final perfomance ? the max after convergence ? an average after convergence ? (but how
#     to quantify convergence ?


def soft_max(values, T):
    v = np.asarray(values)
    # to avoid numerical instabilities
    z = v - np.max(v)
    exp_z = np.exp(z / T)
    probas = exp_z / np.sum(exp_z)
    return probas


class BoltzmannPolicy:
    def __init__(self, temperature=0.005):
        self.T = temperature

    def __call__(self, values):
        probas = self.get_probas(values)
        action = np.random.choice(len(probas), size=1, p=probas)[0]
        return action

    def get_probas(self, values):
        return soft_max(values, self.T)

    def optimal(self, values):
        return np.argmax(values)


class EpsilonGreedyPolicy:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def __call__(self, values):
        if np.random.uniform(0., 1.) < 1. - self.epsilon:
            return np.argmax(values)
        else:
            return np.random.choice(len(values), size=1)[0]

    def optimal(self, values):
        return np.argmax(values)


class TabularAlgo:
    def __init__(self, policy, n_states, n_actions, lr, gamma, verbose):
        self.q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.policy = policy
        self.gamma = gamma
        self.verbose = verbose

    def act(self, obs):
        return self.policy(self.q[obs])

    def optimal_act(self, obs):
        return self.policy.optimal(self.q[obs])

    def episode_learning(self, obs, action, cumul_r, env):
        raise NotImplementedError

    def train(self, env, n_seg=100, ep_per_seg=10):
        seg_eval_rewards = []
        seg_train_rewards = []
        for seg in range(n_seg):
            # training episodes
            for e in range(ep_per_seg):
                obs = env.reset()
                action = self.act(obs)
                done = False
                cumul_r = 0
                while not done:
                    obs, action, done, cumul_r = self.episode_learning(obs, action, cumul_r, env)
                    if self.verbose:
                        print(f'Segment: {seg}, Episode: {e}, Cumul Reward: {round(cumul_r, 4)}')
                seg_train_rewards.append(cumul_r)
            # evaluating episode
            obs = env.reset()
            # env.render()
            done = False
            cumul_r = 0
            while not done:
                action = self.optimal_act(obs)
                obs, r, done, info = env.step(action)
                cumul_r += r
            seg_eval_rewards.append(cumul_r)
            if self.verbose:
                print(f'Segment: {seg}, Testing Cumul Reward: {round(cumul_r, 4)}')

        return seg_eval_rewards, seg_train_rewards


class TabularSarsa(TabularAlgo):
    def __init__(self, policy, n_states, n_actions, lr, gamma, verbose):
        super().__init__(policy, n_states, n_actions, lr, gamma, verbose)

    def episode_learning(self, obs, action, cumul_r, env):
        obs_p, r, done, info = env.step(action)
        action_p = self.act(obs_p)
        self.q[obs, action] = self.q[obs, action] + self.lr * \
                              (r + self.gamma * self.q[obs_p, action_p] - self.q[obs, action])
        cumul_r += r
        return obs_p, action_p, done, cumul_r


class TabularQlearning(TabularAlgo):
    def __init__(self, policy, n_states, n_actions, lr, gamma, verbose):
        super().__init__(policy, n_states, n_actions, lr, gamma, verbose)

    def episode_learning(self, obs, action, cumul_r, env):
        obs_p, r, done, info = env.step(action)
        self.q[obs, action] = self.q[obs, action] + self.lr * \
                              (r + self.gamma * np.max(self.q[obs_p]) - self.q[obs, action])
        cumul_r += r
        action_p = self.act(obs_p)
        return obs_p, action_p, done, cumul_r


class TabularExpectedSarsa(TabularAlgo):
    def __init__(self, policy, n_states, n_actions, lr, gamma, verbose):
        super().__init__(policy, n_states, n_actions, lr, gamma, verbose)

    def episode_learning(self, obs, action, cumul_r, env):
        obs_p, r, done, info = env.step(action)
        expected_q_value = np.sum(self.policy.get_probas(self.q[obs_p]) * self.q[obs_p])
        self.q[obs, action] = self.q[obs, action] + self.lr * \
                              (r + self.gamma * expected_q_value - self.q[obs, action])
        cumul_r += r
        action_p = self.act(obs_p)
        return obs_p, action_p, done, cumul_r


def get_algo(algo_type, policy, n_states, n_actions, lr=0.25, gamma=1, verbose=False):
    if algo_type == 'sarsa':
        algo = TabularSarsa(policy, n_states, n_actions, lr, gamma, verbose)
    elif algo_type == 'Qlearning':
        algo = TabularQlearning(policy, n_states, n_actions, lr, gamma, verbose)
    elif algo_type == 'expected_sarsa':
        algo = TabularExpectedSarsa(policy, n_states, n_actions, lr, gamma, verbose)
    else:
        raise ValueError
    return algo


if __name__ == '__main__':
    import gym
    import matplotlib.pyplot as plt

    env = gym.make("Taxi-v2")
    policy_type = 'softmax'
    algo_type = 'expected_sarsa'

    if policy_type == 'softmax':
        policy = BoltzmannPolicy()
    elif policy_type == 'e_greedy':
        policy = EpsilonGreedyPolicy()

    algo = get_algo(algo_type, policy, env.observation_space.n, env.action_space.n, lr=0.25)

    eval_cumul_r, train_cumul_r = algo.train(env, ep_per_seg=10)

    plt.plot(eval_cumul_r)
    plt.plot(train_cumul_r)
    plt.show()
