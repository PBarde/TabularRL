import gym
import pickle
from algos import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make("Taxi-v2")
    policy_type = 'softmax'
    algo_type = 'expected_sarsa'

    run_eval, run_train = [], []
    for _ in range(10):
        policy = BoltzmannPolicy()
        algo = get_algo(algo_type, policy, env.observation_space.n, env.action_space.n, lr=0.25)
        eval_cumul_r, train_cumul_r = algo.train(env, ep_per_seg=10)
        run_eval.append(eval_cumul_r)
        run_train.append(train_cumul_r)

    m_train = np.mean(run_train, axis=0)
    m_eval = np.mean(run_eval, axis=0)

    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(m_eval)
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Cumulative reward')
    ax1.set_title('Evaluation Performance')
    ax2.plot(m_train)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Cumulative reward')
    ax2.set_title('Training performance')
    plt.tight_layout()
    plt.show()
