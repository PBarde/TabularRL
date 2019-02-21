import numpy as np


# todo : ask questions
# 1 - what is gamma ?
# 2 - do we evaluate always from same initial state ?


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
        probas = soft_max(values, self.T)
        action = np.random.choice(len(probas), size=1, p=probas)[0]
        return action

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
    def __init__(self, policy, n_states, n_actions, lr=0.5, gamma=1):
        self.q = np.zeros((n_states, n_actions))
        self.lr = lr
        self.policy = policy
        self.gamma = gamma

    def act(self, obs):
        return self.policy(self.q[obs])

    def optimal_act(self, obs):
        return self.policy.optimal(self.q[obs])

    def train(self, env, n_seg=100, ep_per_seg=10):
        seg_rewards = []
        for seg in range(n_seg):
            # training episodes
            for e in range(ep_per_seg):
                obs = env.reset()
                action = self.act(obs)
                done = False
                cumul_r = 0
                while not done:
                    obs_p, r, done, info = env.step(action)
                    action_p = self.act(obs_p)
                    self.q[obs, action] = self.q[obs, action] + self.lr * \
                                          (r + self.gamma * self.q[obs_p, action_p] - self.q[obs, action])
                    obs = obs_p
                    action = action_p
                    cumul_r += r
                print(f'Segment: {seg}, Episode: {e}, Cumul Reward: {round(cumul_r, 4)}')

            # evaluating episode
            obs = env.reset()
            # env.render()
            done = False
            cumul_r = 0
            while not done:
                action = self.optimal_act(obs)
                obs, r, done, info = env.step(action)
                cumul_r += r
            seg_rewards.append(cumul_r)
            print(f'Segment: {seg}, Testing Cumul Reward: {round(cumul_r, 4)}')

        return seg_rewards


if __name__ == '__main__':
    import gym
    import matplotlib.pyplot as plt

    env = gym.make("Taxi-v2")
    policy = BoltzmannPolicy()
    # policy = EpsilonGreedyPolicy()
    algo = TabularAlgo(policy, env.observation_space.n, env.action_space.n, lr=0.25)
    eval_cumul_r = algo.train(env, ep_per_seg=100)

    plt.plot(eval_cumul_r)
    plt.show()
