import gym
import pickle
from algos import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make("Taxi-v2")
    policy_type = 'softmax'
    algo_type = 'Qlearning'  # ['expected_sarsa', 'Qlearning', 'sarsa']
    lr_list = [0.25, 0.5, 0.75]
    temperature_list = [1, 10, 100]
    eval_perfs = {}
    train_perfs = {}
    re_eval  = False
    for temp in temperature_list:
        eval_perfs[temp] = {}
        train_perfs[temp] = {}
        for lr in lr_list:
            run_eval, run_train = [], []
            for _ in range(10):
                policy = BoltzmannPolicy(temperature=temp)
                algo = get_algo(algo_type, policy, env.observation_space.n, env.action_space.n, lr=lr)
                eval_cumul_r, train_cumul_r = algo.train(env, ep_per_seg=10)
                # train performance on last 10 episodes
                run_train.append(np.mean(train_cumul_r[-10:]))
                if not re_eval:
                    run_eval.append(eval_cumul_r[-1])
                else:
                    eval_cumul_r = []
                    for ee in range(10):
                        cumul_r = algo.evaluate(env)
                        eval_cumul_r.append(cumul_r)
                    run_eval.append(np.mean (eval_cumul_r))

            eval_perfs[temp].update({lr: np.mean(run_eval)})
            train_perfs[temp].update({lr: np.mean(run_train)})

    f, (ax1, ax2) = plt.subplots(2, 1, sharex='all')

    for i, t in enumerate(temperature_list):
        ax1.plot(lr_list, list(eval_perfs[t].values()), label=f'temperature : {t}', color=f'C{i}')


    ax1.set_ylabel('Cumulative reward')
    ax1.set_title('Evaluation Performance')
    ax1.legend()

    for i, t in enumerate(temperature_list):
        ax2.plot(lr_list, list(train_perfs[t].values()), label=f'temperature : {t}', color=f'C{i}')
    ax2.set_xlabel('Learning rate')
    ax2.set_ylabel('Cumulative reward')
    ax2.set_title('Training performance')
    st = f.suptitle(f'Parameter Effect - {algo_type}')
    plt.tight_layout()
    st.set_y(0.95)
    f.subplots_adjust(top=0.85)
    plt.show()
