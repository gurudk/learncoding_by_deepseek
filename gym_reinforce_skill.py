import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from collections import deque
import random

# 超参数
SKILL_DIM = 4  # 技能维度（隐变量z的大小）
HIDDEN_DIM = 256  # 神经网络隐藏层维度
LR = 3e-4  # 学习率
GAMMA = 0.99  # 折扣因子
BATCH_SIZE = 128  # 批量大小
BUFFER_SIZE = 100000  # 经验回放缓冲区大小
MAX_EPISODE_STEPS = 1000
NUM_EPISODES = 5000

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------- 网络定义 -------------------------
class SkillPolicy(nn.Module):
    """技能策略网络：输入状态s和技能z，输出动作a"""

    def __init__(self, state_dim, action_dim, skill_dim):
        super(SkillPolicy, self).__init__()
        self.skill_dim = skill_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim + skill_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.mu = nn.Linear(HIDDEN_DIM, action_dim)
        self.log_std = nn.Linear(HIDDEN_DIM, action_dim)

    def forward(self, state, skill):
        x = torch.cat([state, skill], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x)
        std = torch.exp(log_std.clamp(-20, 2))
        return torch.distributions.Normal(mu, std)


class SkillDiscriminator(nn.Module):
    """技能判别网络：输入状态s，预测技能z的概率分布"""

    def __init__(self, state_dim, skill_dim):
        super(SkillDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, skill_dim)
        )

    def forward(self, state):
        return self.net(state)


# ------------------------- 经验回放缓冲区 -------------------------
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, next_state, skill, done):
        self.buffer.append((state, action, next_state, skill, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, next_states, skills, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.FloatTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(skills)).to(device),
            torch.FloatTensor(np.array(dones)).to(device),
        )

    def __len__(self):
        return len(self.buffer)


# ------------------------- 训练逻辑 -------------------------
def train():
    # 初始化环境
    env = gym.make("HalfCheetah-v4")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 初始化网络和优化器
    policy = SkillPolicy(state_dim, action_dim, SKILL_DIM).to(device)
    discriminator = SkillDiscriminator(state_dim, SKILL_DIM).to(device)
    optimizer_policy = optim.Adam(policy.parameters(), lr=LR)
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    # 训练循环
    for episode in range(NUM_EPISODES):
        # 随机采样一个技能z（one-hot编码）
        skill = np.eye(SKILL_DIM)[np.random.choice(SKILL_DIM)]
        state, _ = env.reset()
        episode_reward = 0

        for step in range(MAX_EPISODE_STEPS):
            # 将state和skill转换为Tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            skill_tensor = torch.FloatTensor(skill).unsqueeze(0).to(device)

            # 采样动作
            with torch.no_grad():
                action_dist = policy(state_tensor, skill_tensor)
                action = action_dist.sample().cpu().numpy().flatten()

            # 执行动作
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward

            # 存储经验
            replay_buffer.add(state, action, next_state, skill, done)
            state = next_state

            # 更新网络
            if len(replay_buffer) > BATCH_SIZE:
                update_networks(policy, discriminator, optimizer_policy, optimizer_disc, replay_buffer)

            if done:
                break

        # 打印训练进度
        if (episode + 1) % 10 == 0:
            print(f"Episode: {episode + 1}, Reward: {episode_reward:.1f}")


# ------------------------- 网络更新函数 -------------------------
def update_networks(policy, discriminator, optimizer_policy, optimizer_disc, buffer):
    # 采样批量数据
    states, actions, next_states, skills, dones = buffer.sample(BATCH_SIZE)
    skills = skills.to(device)

    # ----------------- 更新判别器 -----------------
    logits = discriminator(states)
    target_z = skills.argmax(dim=1)
    loss_disc = F.cross_entropy(logits, target_z)

    optimizer_disc.zero_grad()
    loss_disc.backward()
    optimizer_disc.step()

    # ----------------- 更新策略 -----------------
    # 计算互信息目标：log q(z|s)
    with torch.no_grad():
        logits = discriminator(states)
        log_q = F.log_softmax(logits, dim=1)
        log_q_z = (log_q * skills).sum(dim=1)

    # 计算策略熵
    action_dist = policy(states, skills)
    log_prob = action_dist.log_prob(actions).sum(dim=1)
    entropy = action_dist.entropy().sum(dim=1)

    # 最大化互信息与熵
    loss_policy = (-log_q_z - 0.1 * entropy).mean()

    optimizer_policy.zero_grad()
    loss_policy.backward()
    optimizer_policy.step()


# ------------------------- 启动训练 -------------------------
if __name__ == "__main__":
    train()
