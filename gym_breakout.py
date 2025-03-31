import  gym

env = gym.make("BreakoutNoFrameskip-v4")
obs = env.reset()

for i in range(1000):
    random_action = env.action_space.sample()
    next_obs, reward, done, info = env.step(random_action)
    env.render()

    if done:
        print("Episode Done")
        env.reset()


