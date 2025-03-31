import numpy as np
import matplotlib.pyplot as plt

# 定义环境：4x4网格，S=起点，G=终点，C=悬崖
grid = [
    ['S', ' ', ' ', ' '],
    [' ', ' ', 'C', ' '],
    [' ', ' ', ' ', 'C'],
    ['C', ' ', ' ', 'G']
]

# 状态空间：每个格子为一个状态（共16个状态）
n_states = 4 * 4
n_actions = 4  # 动作：0=上，1=右，2=下，3=左

# 初始化Q表
Q = np.zeros((n_states, n_actions))

# 超参数
alpha = 0.1  # 学习率
gamma = 0.99  # 折扣因子
epsilon = 0.1  # 探索率
episodes = 500  # 训练轮数


# 状态转换函数
def get_next_state(state, action):
    row, col = state // 4, state % 4
    if action == 0:
        row = max(row - 1, 0)  # 上
    elif action == 1:
        col = min(col + 1, 3)  # 右
    elif action == 2:
        row = min(row + 1, 3)  # 下
    elif action == 3:
        col = max(col - 1, 0)  # 左
    return row * 4 + col


# 奖励函数
def get_reward(state):
    row, col = state // 4, state % 4
    if grid[row][col] == 'G': return 10  # 到达终点
    if grid[row][col] == 'C': return -100  # 掉下悬崖
    return -1  # 普通移动惩罚

try_times = 0

# Q-learning训练
for episode in range(episodes):
    state = 0  # 初始状态（S在左上角）
    done = False

    while not done:
        # ε-greedy选择动作
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(0, 4)  # 随机探索
        else:
            action = np.argmax(Q[state, :])  # 选择最优动作

        # 执行动作，获得下一个状态和奖励
        next_state = get_next_state(state, action)
        reward = get_reward(next_state)

        # 更新Q值
        Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

        # 检查是否终止
        if grid[next_state // 4][next_state % 4] in ['G', 'C']:
            done = True
        state = next_state

        try_times += 1

# 测试训练后的策略
state = 0
path =[state]
total_reward = 0

while True:
    action = np.argmax(Q[state, :])
    next_state = get_next_state(state, action)
    reward = get_reward(next_state)
    total_reward += reward
    path.append(next_state)

    if grid[next_state // 4][next_state % 4] in ['G', 'C']:
        break
    state = next_state

print("最优路径状态序列:", path)
print("总奖励:", total_reward)
print("总尝试次数：", try_times)

# plt.imshow(Q, cmap='hot', interpolation='nearest')
# plt.xlabel('Actions')
# plt.ylabel('States')
# plt.colorbar()
# plt.show()