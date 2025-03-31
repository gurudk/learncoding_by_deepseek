import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 超参数设置
GAMMA = 0.99  # 折扣因子
LR_ACTOR = 0.001  # Actor学习率
LR_CRITIC = 0.005  # Critic学习率
HIDDEN_SIZE = 128  # 神经网络隐藏层大小
MAX_EPISODES = 500  # 最大训练回合数

# 创建CartPole环境
env = gym.make('CartPole-v1')


# 定义Actor网络（策略网络）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        action_probs = F.softmax(self.fc2(x), dim=-1)
        return action_probs


# 定义Critic网络（值函数网络）
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        state_value = self.fc2(x)
        return state_value


# 定义Actor-Critic智能体
class ACAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

    def select_action(self, state):
        # 将状态转换为Tensor
        state = torch.FloatTensor(state)
        # 通过Actor网络获取动作概率
        action_probs = self.actor(state)
        # 按概率分布采样动作
        action = torch.multinomial(action_probs, 1).item()
        return action

    def update(self, state, action, reward, next_state, done):
        # 转换为Tensor
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor([reward])

        # --- Critic更新 ---
        # 计算当前状态的值V(s)和下一状态的值V(s')
        current_value = self.critic(state)
        next_value = self.critic(next_state).detach() * (1 - done)
        # 计算TD目标值
        td_target = reward + GAMMA * next_value
        # 计算Critic损失（均方误差）
        critic_loss = F.mse_loss(current_value, td_target)

        # 反向传播优化Critic
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()

        # --- Actor更新 ---
        # 计算优势函数（TD误差）
        td_error = td_target - current_value.detach()
        # 计算动作的对数概率
        action_probs = self.actor(state)
        log_prob = torch.log(action_probs[action])
        # 计算Actor损失（策略梯度）
        actor_loss = -log_prob * td_error

        # 反向传播优化Actor
        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()


# 初始化环境和智能体
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = ACAgent(state_dim, action_dim)

# 训练循环
episode_rewards = []
for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # 更新网络参数
        agent.update(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

    episode_rewards.append(total_reward)
    print(f"Episode {episode + 1}, Reward: {total_reward}")

    # 如果最近10轮平均奖励大于195，任务解决
    if np.mean(episode_rewards[-10:]) >= 195:
        print("Solved!")
        break

# 绘制奖励曲线
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Actor-Critic Training on CartPole-v1')
plt.show()
