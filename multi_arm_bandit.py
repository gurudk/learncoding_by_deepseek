import numpy as np
import matplotlib.pyplot as plt


# ========== 定义多臂赌博机环境 ==========
class MultiArmedBandit:
    def __init__(self, num_arms=3):
        # 每个臂的真实奖励分布（均值和标准差）
        self.means = np.array([1.0, 2.0, 3.0])  # 假设3个臂的真实均值不同
        self.std_devs = np.array([1.0, 1.0, 1.0])  # 标准差相同

    def pull(self, arm):
        """拉动手臂，返回随机奖励（服从正态分布）"""
        return np.random.normal(self.means[arm], self.std_devs[arm])


# ========== ε-Greedy 算法 ==========
def epsilon_greedy(bandit, num_steps=1000, epsilon=0.1):
    num_arms = len(bandit.means)
    q_values = np.zeros(num_arms)  # 每个臂的预估平均奖励
    counts = np.zeros(num_arms)  # 每个臂被选择的次数
    rewards = []  # 记录每一步的奖励
    optimal_counts = []  # 记录是否选择了最优臂

    optimal_arm = np.argmax(bandit.means)  # 理论最优臂的索引（已知真实均值）

    for step in range(num_steps):
        # 以概率 ε 随机探索，否则利用当前最优
        if np.random.rand() < epsilon:
            chosen_arm = np.random.randint(num_arms)
        else:
            chosen_arm = np.argmax(q_values)

        # 获取奖励并更新统计
        reward = bandit.pull(chosen_arm)
        rewards.append(reward)
        counts[chosen_arm] += 1
        q_values[chosen_arm] += (reward - q_values[chosen_arm]) / counts[chosen_arm]

        # 记录是否选择了最优臂
        optimal_counts.append(1 if chosen_arm == optimal_arm else 0)

    return rewards, optimal_counts


# ========== 运行模拟并可视化 ==========
if __name__ == "__main__":
    np.random.seed(42)  # 固定随机种子，便于复现
    bandit = MultiArmedBandit()
    rewards, optimal_choices = epsilon_greedy(bandit, num_steps=2000, epsilon=0.1)

    # 绘制累计奖励曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(rewards), label='ε=0.1')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward over Time')
    plt.legend()

    # 绘制选择最优臂的频率
    plt.subplot(1, 2, 2)
    optimal_rate = np.cumsum(optimal_choices) / (np.arange(len(optimal_choices)) + 1)
    plt.plot(optimal_rate)
    plt.xlabel('Steps')
    plt.ylabel('Optimal Arm Rate')
    plt.title('Percentage of Choosing Optimal Arm')
    plt.tight_layout()
    plt.show()