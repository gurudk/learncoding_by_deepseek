import numpy as np
import matplotlib.pyplot as plt

def inverse_transform_exponential(lambd, size=1000):
    u = np.random.uniform(0, 1, size)
    return -np.log(1 - u) / lambd

# 生成样本
samples = inverse_transform_exponential(lambd=0.5, size=10000)

# 可视化
plt.hist(samples, bins=50, density=True, alpha=0.6, label='Samples')
x = np.linspace(0, 10, 1000)
plt.plot(x, 0.5 * np.exp(-0.5 * x), 'r', label='True PDF')
plt.legend()
plt.show()