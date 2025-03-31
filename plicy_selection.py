import numpy as np

# 悬崖行走环境：4x12网格，S为起点，G为终点，C为悬崖
# 状态编号：0~47（从左到右，从上到下）
# 动作编号：0=上，1=右，2=下，3=左
class CliffWalkingEnv:
    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start = (3, 0)      # 起点位置
        self.goal = (3, 11)      # 终点位置
        self.current_state = None

    def reset(self):
        self.current_state = self.start
        return self.pos_to_state(self.current_state)

    def pos_to_state(self, pos):
        return pos[0] * self.cols + pos[1]

    def step(self, action):
        row, col = self.current_state
        # 定义动作移动
        if action == 0:     # 上
            row = max(row - 1, 0)
        elif action == 1:   # 右
            col = min(col + 1, self.cols - 1)
        elif action == 2:   # 下
            row = min(row + 1, self.rows - 1)
        elif action == 3:   # 左
            col = max(col - 1, 0)
        # 判断是否掉下悬崖或到达终点
        next_state = (row, col)
        if next_state == self.goal:
            reward = 0
            done = True
        elif row == 3 and 1 <= col <= 10:  # 悬崖区域
            reward = -100
            done = True
        else:
            reward = -1     # 每走一步的惩罚
            done = False
        # 更新状态
        self.current_state = next_state if not done else None
        return self.pos_to_state(next_state), reward, done

def sarsa(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    n_states = env.rows * env.cols
    n_actions = 4
    Q = np.zeros((n_states, n_actions))  # 初始化Q表

    for _ in range(episodes):
        state = env.reset()
        # 初始动作选择（ε-greedy）
        if np.random.rand() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(Q[state])

        done = False
        while not done:
            next_state, reward, done = env.step(action)
            # 选择下一个动作（ε-greedy）
            if np.random.rand() < epsilon:
                next_action = np.random.randint(4)
            else:
                next_action = np.argmax(Q[next_state])
            # SARSA更新公式
            if not done:
                Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            else:
                Q[state][action] += alpha * (reward - Q[state][action])
            # 更新状态和动作
            state, action = next_state, next_action

    return Q

def q_learning(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.1):
    n_states = env.rows * env.cols
    n_actions = 4
    Q = np.zeros((n_states, n_actions))  # 初始化Q表

    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            # 选择动作（ε-greedy）
            if np.random.rand() < epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(Q[state])
            # 执行动作，得到下一个状态和奖励
            next_state, reward, done = env.step(action)
            # Q-learning更新公式（使用max）
            if not done:
                Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
            else:
                Q[state][action] += alpha * (reward - Q[state][action])
            # 更新状态
            state = next_state

    return Q


def print_policy(Q):
    actions = ['↑', '→', '↓', '←']
    for row in range(4):
        line = ''
        for col in range(12):
            state = row * 12 + col
            if (row, col) == (3, 0):
                line += ' S '
            elif (row, col) == (3, 11):
                line += ' G '
            elif row == 3 and 1 <= col <= 10:
                line += ' C '
            else:
                action = np.argmax(Q[state])
                line += f' {actions[action]} '
        print(line)

# 创建环境
env = CliffWalkingEnv()

# 训练并输出SARSA策略
print("SARSA策略（安全路径）：")
Q_sarsa = sarsa(env)
print_policy(Q_sarsa)

# 训练并输出Q-learning策略
print("\nQ-learning策略（冒险路径）：")
Q_qlearning = q_learning(env)
print_policy(Q_qlearning)

