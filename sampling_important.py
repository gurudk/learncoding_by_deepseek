import numpy as np

# 目标分布的概率密度函数（PDF），这里以标准正态分布为例
def p_pdf(x):
    return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

# 建议分布的PDF，这里使用均匀分布U(-5, 5)
def q_pdf(x):
    return np.where((x >= -5) & (x <= 5), 1/10, 0)

# 生成样本
N = 10000
samples = np.random.uniform(-5, 5, N)

# 计算权重
weights = p_pdf(samples) / q_pdf(samples)

# 计算重要性采样估计的期望值
h_values = samples**2
estimate = np.mean(h_values * weights)

print("Importance Sampling Estimate: ", estimate)