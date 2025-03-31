import gymnasium as gym

from ale_py import ALEInterface
ale = ALEInterface()

# 创建 Pong 环境，启用人类可视化模式
env = gym.make(
    "CartPole-v0",
    render_mode="human"
)

# 初始化环境
obs, info = env.reset()

# 运行 1000 步随机策略
for step in range(1000):
    action = env.action_space.sample()  # 随机选择动作
    next_obs, reward, terminated, truncated, info = env.step(action)

    # 重置环境如果游戏结束或截断
    if terminated or truncated:
        next_obs, info = env.reset()

    obs = next_obs

# 关闭环境
env.close()