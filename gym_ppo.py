import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# 超参数设置
ENV_NAME = "CartPole-v1"
GAMMA = 0.99  # 折扣因子
LAMBDA = 0.95  # GAE的衰减系数
EPSILON = 0.2  # PPO裁剪阈值
ACTOR_LR = 3e-4  # Actor学习率
CRITIC_LR = 1e-3  # Critic学习率
BATCH_SIZE = 64  # 小批量大小
MINIBATCH_SIZE = 32  # 子批量大小
EPOCHS = 4  # 更新轮数
MAX_EPISODES = 1000  # 最大训练回合数
MAX_STEPS = 200  # 每回合最大步数
CLIP_GRAD = 0.5  # 梯度裁剪阈值

# 创建环境
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n


# 定义Actor网络（策略网络）
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        logits = self.fc2(x)
        return logits  # 输出动作的logits（未归一化）

    def get_action(self, state):
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


# 定义Critic网络（值函数网络）
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        value = self.fc2(x)
        return value


# 定义PPO智能体
class PPOAgent:
    def __init__(self):
        self.actor = Actor()
        self.critic = Critic()
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.memory = deque(maxlen=BATCH_SIZE)  # 经验回放缓冲区

    def store_transition(self, state, action, reward, next_state, done, old_log_prob):
        self.memory.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "old_log_prob": old_log_prob,
        })

    def compute_gae(self, rewards, dones, values, next_values):
        # 计算广义优势估计（GAE）
        deltas = [r + GAMMA * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        advantages = []
        advantage = 0.0
        for delta in reversed(deltas):
            advantage = delta + GAMMA * LAMBDA * advantage
            advantages.insert(0, advantage)
        return torch.tensor(advantages, dtype=torch.float32)

    def update(self):
        # 从经验回放中提取数据
        states = torch.tensor([t["state"] for t in self.memory], dtype=torch.float32)
        actions = torch.tensor([t["action"] for t in self.memory], dtype=torch.int64)
        old_log_probs = torch.tensor([t["old_log_prob"] for t in self.memory], dtype=torch.float32)
        rewards = torch.tensor([t["reward"] for t in self.memory], dtype=torch.float32)
        next_states = torch.tensor([t["next_state"] for t in self.memory], dtype=torch.float32)
        dones = torch.tensor([t["done"] for t in self.memory], dtype=torch.float32)

        # 计算值函数和GAE
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
        advantages = self.compute_gae(rewards, dones, values, next_values)
        returns = advantages + values

        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 多轮次更新（PPO的核心）
        for _ in range(EPOCHS):
            # 随机打乱数据并分批次
            indices = np.arange(len(self.memory))
            np.random.shuffle(indices)
            for start in range(0, len(self.memory), MINIBATCH_SIZE):
                idx = indices[start: start + MINIBATCH_SIZE]
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                # 计算新的动作概率
                logits = self.actor(batch_states)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)

                # 重要性采样比率
                ratios = torch.exp(new_log_probs - batch_old_log_probs)

                # 裁剪的PPO损失
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - EPSILON, 1 + EPSILON) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic损失（值函数均方误差）
                current_values = self.critic(batch_states).squeeze()
                critic_loss = F.mse_loss(current_values, batch_returns)

                # 总损失（可选：加入熵正则化）
                entropy = dist.entropy().mean()
                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                # 反向传播和优化
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), CLIP_GRAD)  # 梯度裁剪
                nn.utils.clip_grad_norm_(self.critic.parameters(), CLIP_GRAD)
                self.actor_optim.step()
                self.critic_optim.step()

        # 清空经验回放
        self.memory.clear()


# 训练函数
def train():
    agent = PPOAgent()
    episode_rewards = []

    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        for _ in range(MAX_STEPS):
            # 收集经验数据
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action, old_log_prob = agent.actor.get_action(state_tensor)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done, old_log_prob)
            state = next_state
            episode_reward += reward

            if done:
                break

        # 经验回放缓冲区填满后更新网络
        if len(agent.memory) >= BATCH_SIZE:
            agent.update()

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}, Reward: {episode_reward}")

        # 提前终止条件
        if np.mean(episode_rewards[-10:]) >= 195:
            print("Solved!")
            break

    # 绘制奖励曲线
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("PPO Training on CartPole-v1")
    plt.show()


if __name__ == "__main__":
    train()
