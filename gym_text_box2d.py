import gymnasium as gym

env = gym.make("BipedalWalker-v3", render_mode="human")

env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())