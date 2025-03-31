import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 生成数据
n_samples = 50
x = np.linspace(0, 10, n_samples)
true_beta0 = 2.0
true_beta1 = 0.5
true_sigma = 1.0

y = true_beta0 + true_beta1 * x + np.random.normal(0, true_sigma, n_samples)

# 可视化数据
plt.scatter(x, y, alpha=0.7, label="观测数据")
plt.plot(x, true_beta0 + true_beta1 * x, 'r--', label="真实回归线")
plt.xlabel("房屋面积 (x)")
plt.ylabel("房价 (y)")
plt.legend()
plt.show()


def log_prior(beta0, beta1, log_sigma):
    """计算对数先验概率"""
    # beta0 ~ N(0,10)
    log_p_beta0 = -0.5 * (beta0 ** 2) / 10 - 0.5 * np.log(10 * 2 * np.pi)
    # beta1 ~ N(0,10)
    log_p_beta1 = -0.5 * (beta1 ** 2) / 10 - 0.5 * np.log(10 * 2 * np.pi)
    # sigma^2 ~ InverseGamma(1,1) => log_sigma的变换后先验
    log_p_sigma = -2 * log_sigma - 1.0 / (np.exp(2 * log_sigma))  # Jacobian变换
    return log_p_beta0 + log_p_beta1 + log_p_sigma


def log_likelihood(beta0, beta1, log_sigma, x, y):
    """计算对数似然"""
    sigma = np.exp(log_sigma)
    y_pred = beta0 + beta1 * x
    log_lik = -0.5 * np.sum((y - y_pred) ** 2) / (sigma ** 2) - n_samples * np.log(sigma)
    return log_lik


def log_posterior(beta0, beta1, log_sigma, x, y):
    """计算对数后验（未归一化）"""
    return log_prior(beta0, beta1, log_sigma) + log_likelihood(beta0, beta1, log_sigma, x, y)


def metropolis_hastings(n_iter, step_sizes, initial_params):
    """MH采样主函数"""
    params = np.zeros((n_iter, 3))
    params[0] = initial_params
    accept = np.zeros(n_iter)

    for t in range(1, n_iter):
        # 当前参数
        current = params[t - 1]
        beta0, beta1, log_sigma = current

        # 生成候选参数（对称正态提议）
        candidate = current + np.random.normal(0, step_sizes)

        # 计算接受概率
        log_alpha = log_posterior(*candidate, x, y) - log_posterior(*current, x, y)
        alpha = min(1, np.exp(log_alpha))

        # 接受或拒绝
        if np.random.rand() < alpha:
            params[t] = candidate
            accept[t] = 1
        else:
            params[t] = current

    accept_rate = np.mean(accept)
    print(f"平均接受率: {accept_rate:.2f}")
    return params


# 参数设置
n_iter = 10000
step_sizes = np.array([0.1, 0.05, 0.1])  # beta0, beta1, log_sigma的步长
initial_params = np.array([0.0, 0.0, 0.0])  # 初始值

# 运行MH采样
samples = metropolis_hastings(n_iter, step_sizes, initial_params)

burn_in = 2000
posterior_samples = samples[burn_in:]
beta0_samples = posterior_samples[:, 0]
beta1_samples = posterior_samples[:, 1]
sigma_samples = np.exp(posterior_samples[:, 2])  # 转换回sigma

# 绘制后验分布直方图
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(beta0_samples, bins=50, density=True, alpha=0.7)
axes[0].axvline(true_beta0, color='r', linestyle='--', label="真实值")
axes[0].set_title(r"$\beta_0$ 后验分布")

axes[1].hist(beta1_samples, bins=50, density=True, alpha=0.7)
axes[1].axvline(true_beta1, color='r', linestyle='--', label="真实值")
axes[1].set_title(r"$\beta_1$ 后验分布")

axes[2].hist(sigma_samples, bins=50, density=True, alpha=0.7)
axes[2].axvline(true_sigma, color='r', linestyle='--', label="真实值")
axes[2].set_title(r"$\sigma$ 后验分布")

plt.tight_layout()
plt.show()


# 绘制参数轨迹
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

axes[0].plot(beta0_samples, alpha=0.7)
axes[0].axhline(true_beta0, color='r', linestyle='--')
axes[0].set_ylabel(r"$\beta_0$")

axes[1].plot(beta1_samples, alpha=0.7)
axes[1].axhline(true_beta1, color='r', linestyle='--')
axes[1].set_ylabel(r"$\beta_1$")

axes[2].plot(sigma_samples, alpha=0.7)
axes[2].axhline(true_sigma, color='r', linestyle='--')
axes[2].set_ylabel(r"$\sigma$")

plt.tight_layout()
plt.show()

# 生成后验预测样本
x_new = np.linspace(0, 10, 100)
y_pred_samples = np.zeros((len(posterior_samples), len(x_new)))

for i in range(len(posterior_samples)):
    beta0 = posterior_samples[i, 0]
    beta1 = posterior_samples[i, 1]
    sigma = np.exp(posterior_samples[i, 2])
    y_pred = beta0 + beta1 * x_new + np.random.normal(0, sigma, len(x_new))
    y_pred_samples[i] = y_pred

# 计算95%置信区间
lower = np.percentile(y_pred_samples, 2.5, axis=0)
upper = np.percentile(y_pred_samples, 97.5, axis=0)
mean = np.mean(y_pred_samples, axis=0)

# 可视化预测区间
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.7, label="观测数据")
plt.plot(x_new, mean, 'r-', label="后验预测均值")
plt.fill_between(x_new, lower, upper, color='red', alpha=0.2, label="95%置信区间")
plt.plot(x_new, true_beta0 + true_beta1 * x_new, 'k--', label="真实回归线")
plt.xlabel("房屋面积 (x)")
plt.ylabel("房价 (y)")
plt.legend()
plt.show()