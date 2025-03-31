import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)


# 超参数
gamma = 0.99
lr = 0.01
max_episodes = 1000

# 初始化环境和策略
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
policy = PolicyNetwork(state_dim, action_dim)
optimizer = optim.Adam(policy.parameters(), lr=lr)

# 训练循环
for episode in range(max_episodes):
    state, info = env.reset()
    states, actions, rewards = [], [], []

    # 采样轨迹（On-policy）
    while True:
        state_tensor = torch.FloatTensor(state)
        action_probs = policy(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample().item()

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = next_state
        if done:
            break

    # 计算折扣回报
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)

    # 归一化回报（减少方差）
    returns = torch.FloatTensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # 计算梯度并更新策略
    policy_loss = []
    for state, action, R in zip(states, actions, returns):
        state_tensor = torch.FloatTensor(state)
        action_probs = policy(state_tensor)
        dist = Categorical(action_probs)
        log_prob = dist.log_prob(torch.tensor(action))
        policy_loss.append(-log_prob * R)

    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

    # 打印训练进度
    if episode % 50 == 0:
        print(f"Episode {episode}, Total Reward: {sum(rewards)}")

env.close()