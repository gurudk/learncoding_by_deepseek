import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 初始化Q表
q_table = np.zeros([16, 4])  # 16个状态，每个状态有4个可能的动作

# 定义动作集：上下左右
actions = ['up', 'down', 'left', 'right']

# 定义一些参数
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1
num_episodes = 2000


def get_next_state(state, action):
    row = state // 4
    col = state % 4
    if actions[action] == 'up' and row > 0:
        row -= 1
    elif actions[action] == 'down' and row < 3:
        row += 1
    elif actions[action] == 'left' and col > 0:
        col -= 1
    elif actions[action] == 'right' and col < 3:
        col += 1
    return row * 4 + col


frames = []
for episode in range(num_episodes):
    state = 0  # 初始状态为左上角
    done = False

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(len(actions))  # 探索
        else:
            action = np.argmax(q_table[state])  # 利用

        next_state = get_next_state(state, action)
        reward = -1 if next_state != 15 else 0  # 到达终点奖励为0，其他情况为-1
        done = next_state == 15

        q_table[state, action] += learning_rate * (
                    reward + discount_factor * np.max(q_table[next_state]) - q_table[state, action])

        frames.append((state, next_state))  # 记录移动轨迹
        state = next_state

# 动画演示部分
fig, ax = plt.subplots()
ax.set_xlim(-1, 4)
ax.set_ylim(-1, 4)
ax.grid(True)

maze_path, = ax.plot([], [], 'o-', lw=2)


def init():
    maze_path.set_data([], [])
    return maze_path,


def animate(i):
    x = [frames[i][0] % 4, frames[i][1] % 4]
    y = [frames[i][0] // 4, frames[i][1] // 4]
    maze_path.set_data(x, y)
    return maze_path,


ani = FuncAnimation(fig, animate, frames=len(frames), init_func=init, blit=True, repeat=False)
plt.gca().invert_yaxis()  # 反转y轴以匹配迷宫坐标系
plt.show()