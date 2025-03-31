import numpy as np

def target_distribution(x):
    # 定义目标分布（示例为标准正态分布）
    return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

def metropolis_hastings(n_samples, initial=0.0, proposal_std=1.0):
    samples = [initial]
    current = initial
    for _ in range(n_samples-1):
        proposal = np.random.normal(current, proposal_std)
        acceptance_ratio = target_distribution(proposal) / target_distribution(current)
        if np.random.rand() < acceptance_ratio:
            current = proposal
        samples.append(current)
    return samples

# 生成样本并绘制分布
samples = metropolis_hastings(10000)
import matplotlib.pyplot as plt
plt.hist(samples, bins=50, density=True)
plt.show()