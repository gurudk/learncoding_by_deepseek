import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import robosuite as suite
from robosuite.wrappers import VisualizationWrapper

# 超参数
STATE_DIM = 10  # 状态维度（根据环境调整）
ACTION_DIM = 7  # 动作维度（机械臂关节控制）
HIDDEN_SIZE = 256
BATCH_SIZE = 128
BUFFER_SIZE = 100000
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
MAX_EPISODE_STEPS = 500
TRAIN_EPISODES = 2000


# 创建Robosuite环境
def make_env():
    env = suite.make(
        "Lift",
        robots="Panda",
        use_camera_obs=False,  # 不使用视觉输入
        has_offscreen_renderer=False,
        use_object_obs=True,
        reward_shaping=True,
    )
    return VisualizationWrapper(env)  # 启用可视化


# SAC Actor网络
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(HIDDEN_SIZE, ACTION_DIM)
        self.log_std_head = nn.Linear(HIDDEN_SIZE, ACTION_DIM)

    def forward(self, state):
        x = self.net(state)
        mu = torch.tanh(self.mu_head(x))  # 输出在[-1,1]范围
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mu, log_std


# SAC Critic网络
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q1(x), self.q2(x)


# SAC智能体
class SACAgent:
    def __init__(self):
        self.env = make_env()
        self.actor = Actor()
        self.critic = Critic()
        self.target_critic = Critic()
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.optim_actor = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.buffer = deque(maxlen=BUFFER_SIZE)
        self.total_steps = 0

    # 选择动作（带探索噪声）
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mu, log_std = self.actor(state)
            std = log_std.exp()
            normal = torch.distributions.Normal(mu, std)
            action = normal.sample()
        return action.squeeze(0).numpy()

    # 存储经验
    def add_experience(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # 训练步骤
    def train_step(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        # 采样批次数据
        batch = np.random.choice(len(self.buffer), BATCH_SIZE, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])

        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # 更新Critic
        with torch.no_grad():
            next_mu, next_log_std = self.actor(next_states)
            next_actions = torch.tanh(next_mu + torch.randn_like(next_log_std) * next_log_std.exp())
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * GAMMA * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()

        # 更新Actor
        mu, log_std = self.actor(states)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        sampled_actions = torch.tanh(mu + torch.randn_like(std) * std)
        log_prob = normal.log_prob(mu).sum(dim=1, keepdim=True)

        q1, q2 = self.critic(states, sampled_actions)
        q = torch.min(q1, q2)
        actor_loss = (ALPHA * log_prob - q).mean()

        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()

        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    # 训练主循环
    def train(self):
        print("开始训练...")
        for ep in range(TRAIN_EPISODES):
            state = self.env.reset()
            total_reward = 0
            done = False
            steps = 0

            while not done and steps < MAX_EPISODE_STEPS:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.add_experience(state, action, reward, next_state, done)
                self.train_step()

                total_reward += reward
                state = next_state
                steps += 1
                self.total_steps += 1

            if ep % 50 == 0:
                print(f"回合 {ep}, 总奖励: {total_reward:.1f}, 步数: {steps}")

    # 可视化测试
    def visualize(self, num_episodes=3):
        print("\n开始可视化测试...")
        test_env = make_env()

        for ep in range(num_episodes):
            state = test_env.reset()
            total_reward = 0
            done = False
            steps = 0

            while not done and steps < MAX_EPISODE_STEPS:
                action = self.select_action(state)
                next_state, reward, done, _ = test_env.step(action)
                test_env.render()

                total_reward += reward
                state = next_state
                steps += 1
                time.sleep(0.02)

            print(f"测试回合 {ep + 1}, 总奖励: {total_reward:.1f}")
        test_env.close()


if __name__ == "__main__":
    agent = SACAgent()
    agent.train()
    agent.visualize()
