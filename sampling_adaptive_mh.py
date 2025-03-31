import numpy as np
import matplotlib.pyplot as plt


# 目标分布：二维高斯分布
def target(x, mu=np.array([0, 0]), Sigma=np.array([[1, 0.8], [0.8, 1]])):
    inv_Sigma = np.linalg.inv(Sigma)
    norm = 1 / (2 * np.pi * np.sqrt(np.linalg.det(Sigma)))
    return norm * np.exp(-0.5 * (x - mu).T @ inv_Sigma @ (x - mu))


# 自适应MH采样
def adaptive_mh(n_samples, initial, initial_cov, adapt_interval=100):
    x = initial
    samples = [x]
    cov = initial_cov
    accept = 0

    for t in range(1, n_samples):
        # 生成候选样本
        candidate = np.random.multivariate_normal(x, cov)

        # 计算接受概率
        alpha = min(1, target(candidate) / target(x))

        # 接受或拒绝
        if np.random.rand() < alpha:
            x = candidate
            accept += 1
        samples.append(x)

        # 每adapt_interval步更新协方差
        if t % adapt_interval == 0:
            samples_array = np.array(samples)
            cov = np.cov(samples_array.T) + 1e-6 * np.eye(2)

    print(f"接受率: {accept / n_samples:.2f}")
    return np.array(samples)


# 参数设置
n_samples = 10000
initial = np.array([-5, 5])
initial_cov = np.eye(2) * 5  # 初始大协方差以鼓励探索

samples = adaptive_mh(n_samples, initial, initial_cov)

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)
plt.title("自适应MH采样结果（二维高斯）")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()