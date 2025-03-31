import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子（保证结果可复现）
np.random.seed(42)


# ==============================
# 1. 定义目标分布（双峰高斯混合）
# ==============================
def target_distribution(x):
    # 两个高斯分布的权重和参数
    peak1_weight = 0.3
    peak2_weight = 0.7
    gaussian1 = np.exp(-0.5 * ((x + 3) / 1) ** 2) / (1 * np.sqrt(2 * np.pi))
    gaussian2 = np.exp(-0.5 * ((x - 2) / 0.5) ** 2) / (0.5 * np.sqrt(2 * np.pi))
    return peak1_weight * gaussian1 + peak2_weight * gaussian2


# ==============================
# 2. Metropolis-Hastings采样
# ==============================
def metropolis_hastings(n_samples, step_size, initial_state):
    samples = []
    current_state = initial_state
    accept_count = 0  # 记录接受次数

    for _ in range(n_samples):
        # 生成候选样本（提议分布为对称的正态分布）
        candidate = np.random.normal(current_state, step_size)

        # 计算接受概率（因提议分布对称，简化为目标分布的比值）
        accept_ratio = target_distribution(candidate) / target_distribution(current_state)
        alpha = min(1, accept_ratio)

        # 决定是否接受候选样本
        if np.random.rand() < alpha:
            current_state = candidate
            accept_count += 1

        samples.append(current_state)

    # 计算接受率
    accept_rate = accept_count / n_samples
    print(f"接受率: {accept_rate:.2f}")
    return np.array(samples)


# ==============================
# 3. 参数设置与采样
# ==============================
n_samples = 10000  # 总采样数
step_size = 1.0  # 提议分布的步长
initial_state = 0.0  # 初始状态

samples = metropolis_hastings(n_samples, step_size, initial_state)

# ==============================
# 4. 可视化结果
# ==============================
# 绘制采样结果直方图
plt.figure(figsize=(12, 5))

# 直方图与真实分布对比
x_range = np.linspace(-6, 5, 1000)
plt.hist(samples, bins=50, density=True, alpha=0.6, label="MH采样结果")
plt.plot(x_range, target_distribution(x_range), 'r', linewidth=2, label="真实分布")
plt.title("目标分布 vs MH采样结果")
plt.xlabel("x")
plt.ylabel("密度")
plt.legend()

# 绘制采样路径（前200步）
plt.figure(figsize=(12, 3))
plt.plot(samples[:200], 'o-', markersize=4, alpha=0.7)
plt.title("采样路径（前200步）")
plt.xlabel("迭代次数")
plt.ylabel("x值")

plt.show()