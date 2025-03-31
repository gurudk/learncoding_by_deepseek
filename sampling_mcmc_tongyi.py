import numpy as np
import matplotlib.pyplot as plt


# 定义目标分布的概率密度函数（这里以标准正态分布为例）
def target_distribution(x):
    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)


# Metropolis-Hastings采样算法
def metropolis_hastings(target, num_samples=10000, burn_in=1000):
    samples = []
    x = 0  # 初始状态可以任意选择

    for i in range(num_samples + burn_in):
        # 提议分布通常选择对称分布，例如正态分布
        x_candidate = np.random.normal(loc=x, scale=1)

        # 计算接受概率
        acceptance_prob = min(1, target(x_candidate) / target(x))

        # 接受或拒绝新状态
        if np.random.rand() < acceptance_prob:
            x = x_candidate

        # 仅在burn-in期后保存样本
        if i >= burn_in:
            samples.append(x)

    return np.array(samples)


# 执行采样
samples = metropolis_hastings(target_distribution)

# 绘制直方图和目标分布的曲线
plt.hist(samples, bins=50, density=True, alpha=0.6, color='g', label='MCMC Samples')

x = np.linspace(-4, 4, 1000)
plt.plot(x, target_distribution(x), 'r', lw=2, label='Target Distribution')
plt.legend()
plt.show()