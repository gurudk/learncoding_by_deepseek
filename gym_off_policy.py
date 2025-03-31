import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


# 定义Q网络
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# 超参数
gamma = 0.99
lr = 0.001
batch_size = 64
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
target_update = 10
max_episodes = 1000
buffer_capacity = 10000

# 初始化环境、网络和优化器
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
q_network = QNetwork(state_dim, action_dim)
target_network = QNetwork(state_dim, action_dim)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=lr)
buffer = ReplayBuffer(buffer_capacity)
epsilon = epsilon_start

# 训练循环
for episode in range(max_episodes):
    state, info = env.reset()
    total_reward = 0

    while True:
        # ε-greedy行为策略（Off-policy）
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state)
            q_values = q_network(state_tensor)
            action = torch.argmax(q_values).item()

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward

        # 存储经验
        buffer.push(state, action, reward, next_state, done)
        state = next_state

        # 更新Q网络
        if len(buffer) >= batch_size:
            batch = buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(dones)

            # 计算目标Q值
            with torch.no_grad():
                target_q = rewards + gamma * (1 - dones) * torch.max(target_network(next_states), dim=1)[0]

            # 计算当前Q值
            current_q = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

            # 计算损失
            loss = nn.MSELoss()(current_q, target_q)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    # 更新目标网络
    if episode % target_update == 0:
        target_network.load_state_dict(q_network.state_dict())

    # 衰减ε
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    # 打印训练进度
    if episode % 50 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

env.close()