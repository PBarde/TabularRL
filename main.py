import gym

env = gym.make("Taxi-v2")
for _ in range(10):
    observation = env.reset()
    print(env.action_space)
    # env.render()
