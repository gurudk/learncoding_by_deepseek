import numpy as np

def target_distribution(x):
    return np.exp(-x)  # 指数分布

def proposal_distribution():
    return np.random.uniform(0, 10)  # 均匀分布作为提议分布

def rejection_sampling(n_samples, M=1.0):
    samples = []
    while len(samples) < n_samples:
        x = proposal_distribution()
        u = np.random.uniform(0, M)
        if u <= target_distribution(x):
            samples.append(x)
    return samples

# 示例
samples = rejection_sampling(1000)
import matplotlib.pyplot as plt
plt.hist(samples, bins=50, density=True)
plt.show()