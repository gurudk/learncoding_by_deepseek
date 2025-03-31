import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# 超参数
NUM_TASKS = 5
META_BATCH_SIZE = 5
INNER_STEPS = 3
INNER_LR = 0.1
META_LR = 1e-3
NUM_EPOCHS = 5000
HIDDEN_DIM = 64
MAX_STEPS = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------- 安全环境 -------------------------
class SafeMultiGoalEnv(gym.Env):
    def __init__(self, num_tasks):
        self.num_tasks = num_tasks
        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(2,))
        self.observation_space = gym.spaces.Box(low=-5, high=5, shape=(4,))
        self.goal = np.zeros(2)
        self.state = np.zeros(4)

    def reset(self):
        self.goal = np.random.uniform(-2, 2, size=2)
        self.state = np.concatenate([np.zeros(2), np.zeros(2)])
        return self.state.copy(), {"task_id": random.randint(0, self.num_tasks - 1)}

    def step(self, action):
        self.state[2:] = np.clip(action, -0.1, 0.1)
        self.state[:2] += self.state[2:]
        self.state[:2] = np.clip(self.state[:2], -5, 5)
        self.state[2:] = np.clip(self.state[2:], -0.5, 0.5)
        distance = np.linalg.norm(self.state[:2] - self.goal)
        reward = -distance
        done = distance < 0.5
        # print(
        #     f"Position: {self.state[:2]}, Goal: {self.goal}, Distance: {np.linalg.norm(self.state[:2] - self.goal):.2f}, Done: {done}")  # 调试输出
        return self.state.copy(), reward, done, False, {"goal": self.goal}


# ------------------------- 稳定策略网络 -------------------------
class StablePolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(StablePolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, HIDDEN_DIM)
        self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, action_dim)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -1.0)  # 初始化为-1

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.fc3(x))
        std = torch.exp(self.log_std.clamp(-5, 2)) + 1e-6  # 添加下限并限制范围
        return torch.distributions.Normal(mean, std)


# ------------------------- 元训练逻辑 -------------------------
def safe_meta_train():
    env = SafeMultiGoalEnv(NUM_TASKS)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    policy = StablePolicy(state_dim, action_dim).to(device)
    meta_optimizer = optim.Adam(policy.parameters(), lr=META_LR)

    for epoch in range(NUM_EPOCHS):
        task_indices = np.random.choice(NUM_TASKS, META_BATCH_SIZE)
        total_loss = 0

        for task in task_indices:
            fast_weights = {k: v.clone() for k, v in policy.named_parameters()}

            # 内循环适应
            for _ in range(INNER_STEPS):
                states, actions, rewards = [], [], []
                state, _ = env.reset()
                done = step_count = 0
                while not done and step_count < MAX_STEPS:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

                    def get_dist(params, x):
                        x = torch.relu(nn.functional.linear(x, params['fc1.weight'], params['fc1.bias']))
                        x = torch.relu(nn.functional.linear(x, params['fc2.weight'], params['fc2.bias']))
                        mean = torch.tanh(nn.functional.linear(x, params['fc3.weight'], params['fc3.bias']))
                        std = torch.exp(params['log_std'].clamp(-5, 2)) + 1e-6  # 同步修改
                        return torch.distributions.Normal(mean, std)

                    with torch.no_grad():
                        dist = get_dist(fast_weights, state_tensor)
                        action = dist.sample().cpu().numpy().flatten()

                    next_state, reward, done, _, _ = env.step(action)
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    state = next_state
                    step_count += 1

                if len(states) > 0:
                    states_tensor = torch.FloatTensor(np.array(states)).to(device)
                    actions_tensor = torch.FloatTensor(np.array(actions)).to(device)
                    returns = torch.FloatTensor([sum(rewards[i:]) for i in range(len(rewards))]).to(device)

                    dist = get_dist(fast_weights, states_tensor)
                    log_probs = dist.log_prob(actions_tensor).sum(dim=1)
                    loss = -(log_probs * returns).mean()

                    grads = torch.autograd.grad(loss, fast_weights.values(),
                                                create_graph=True, allow_unused=True)
                    grads = [g if g is not None else torch.zeros_like(p)
                             for g, p in zip(grads, fast_weights.values())]
                    fast_weights = {k: v - INNER_LR * g for (k, v), g in zip(fast_weights.items(), grads)}

            # 元梯度计算
            state, _ = env.reset()
            done = step_count = 0
            meta_states, meta_actions = [], []
            while not done and step_count < MAX_STEPS:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    dist = get_dist(fast_weights, state_tensor)
                    action = dist.sample().cpu().numpy().flatten()
                next_state, _, done, _, _ = env.step(action)
                meta_states.append(state)
                meta_actions.append(action)
                state = next_state
                step_count += 1

            if len(meta_states) > 0:
                meta_states_tensor = torch.FloatTensor(np.array(meta_states)).to(device)
                meta_actions_tensor = torch.FloatTensor(np.array(meta_actions)).to(device)
                dist = policy(meta_states_tensor)
                log_probs = dist.log_prob(meta_actions_tensor).sum(dim=1)
                meta_loss = -log_probs.mean()
                total_loss += meta_loss.item()

                meta_optimizer.zero_grad()
                meta_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)  # 更严格的梯度裁剪
                meta_optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}, Loss: {total_loss / META_BATCH_SIZE:.2f}")


if __name__ == "__main__":
    safe_meta_train()
