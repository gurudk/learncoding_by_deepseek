import numpy as np

def target_distribution(x):
    return np.exp(-x**2 / 2)  # 目标分布（未归一化）

def proposal_distribution(x):
    return np.exp(-(x-1)**2 / 2)  # 提议分布（均值偏移的正态分布）

def importance_sampling(n_samples):
    samples = np.random.normal(1, 1, n_samples)  # 从提议分布采样
    weights = [target_distribution(x)/proposal_distribution(x) for x in samples]
    return samples, weights

# 示例
samples, weights = importance_sampling(1000)
# print(len(samples), len(weights))
fsamples = samples * weights
print(fsamples)
import matplotlib.pyplot as plt
plt.hist(fsamples, bins=50, density=True)
plt.show()
