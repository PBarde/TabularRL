import gym
import pickle
from algos import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make("Taxi-v2")
    policy_type = 'softmax'
    algo_type = 'expected_sarsa'
    lr = 0.75
    temperature = 10
    run_eval, run_train = [], []
    for _ in range(10):
        policy = BoltzmannPolicy(temperature)
        algo = get_algo(algo_type, policy, env.observation_space.n, env.action_space.n, lr=lr)
        eval_cumul_r, train_cumul_r = algo.train(env, ep_per_seg=10)
        run_eval.append(eval_cumul_r)
        run_train.append(train_cumul_r)

    m_train = np.mean(run_train, axis=0)
    m_eval = np.mean(run_eval, axis=0)

    std_train = np.std(run_train, axis=0)
    std_eval = np.std(run_eval, axis=0)

    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(m_eval, color='C1')
    ax1.fill_between(range(len(m_eval)), m_eval - std_eval, m_eval + std_eval, facecolor='C1', alpha=0.5)
    ax1.set_xlabel('Evaluation Episodes')
    ax1.set_ylabel('Cumulative reward')
    ax1.set_title('Evaluation Performance')
    ax2.plot(m_train, color='C2')
    ax2.fill_between(range(len(m_train)), m_train - std_train, m_train + std_train, facecolor='C2', alpha=0.5)
    ax2.set_xlabel('Training Iterations')
    ax2.set_ylabel('Cumulative reward')
    ax2.set_title('Training performance')
    st = f.suptitle(f' Training curve - {algo_type}')
    plt.tight_layout()
    st.set_y(0.95)
    f.subplots_adjust(top=0.85)
    plt.show()
