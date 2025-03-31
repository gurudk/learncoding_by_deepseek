import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

ENV_NAME = "CartPole-v1"
GAMMA = 0.99
LR = 0.01
MAX_EPISODES = 1000

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = PolicyNet(state_dim, action_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=LR)

for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    rewards = []
    log_probs = []

    while True:
        state_tensor = torch.FloatTensor(state)
        action_probs = policy_net(state_tensor)
        m = Categorical(action_probs)
        action = m.sample()
        log_prob = m.log_prob(action)

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        log_probs.append(log_prob)
        rewards.append(reward)
        state = next_state

        if done:
            break

    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + GAMMA * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)

    # 修复点：使用 torch.stack 或 sum()
    policy_loss = []
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)
    policy_loss = torch.stack(policy_loss).sum()  # 或 policy_loss = sum(policy_loss)

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()

    if episode % 50 == 0:
        print(f"Episode {episode}, Total Reward: {sum(rewards)}")

env.close()