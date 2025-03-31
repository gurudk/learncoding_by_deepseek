import gymnasium as gym

from ale_py import ALEInterface
from conda.reporters import render
from pyro import render_model

ale = ALEInterface()

env = gym.make("ALE/Qbert-v5",
                render_mode="human",  # 或 "rgb_array" 获取像素
                obs_type="rgb"  # 观测类型：可选 "rgb" (图像) 或 "grayscale"
               )
MAX_NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 500
for episode in range(MAX_NUM_EPISODES):
    obs = env.reset()
    for step in range(MAX_STEPS_PER_EPISODE):
        env.render()
        action = env.action_space.sample()  # Sample random action. This will be replaced by our agent's action when we start developing the agent algorithms
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        obs = observation

        if done is True:
            print("\n Episode #{} ended in {} steps.".format(episode, step + 1))
            break

