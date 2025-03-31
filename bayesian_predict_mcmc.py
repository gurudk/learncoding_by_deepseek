import numpy as np
import matplotlib.pyplot as plt

# 生成模拟数据
np.random.seed(42)
mu_true = 3.0
sigma = 1.0
n = 500
y = np.random.normal(mu_true, sigma, n)

# Metropolis-Hastings 参数
n_samples = 10000
burn_in = 2000
mu_current = 0
proposal_sd = 0.5
samples = []

# 对数后验函数（忽略常数项）
def log_posterior(mu, y, mu_0, tau_0):
    log_prior = -0.5*(mu-mu_0)**2/tau_0**2
    log_likelihood = -0.5*np.sum((y-mu)**2)/sigma**2
    return log_prior + log_likelihood

# MCMC采样
for _ in range(n_samples + burn_in):
    # 提议新参数,从当前分布里采样一个值，用作建议值
    mu_proposal = np.random.normal(mu_current, proposal_sd)

    # 计算接受概率
    log_alpha= log_posterior(mu_proposal, y, mu_0=0.0, tau_0=2.0) - log_posterior(mu_current, y, mu_0=0.0,tau_0=2.0)
    alpha = np.exp(log_alpha)

    if np.random.rand() < alpha:
        mu_current = mu_proposal
    samples.append(mu_current)


# 去除 burn-in阶段
samples = samples[burn_in:]

# 生成后验预测样本
y_new_samples = [np.random.normal(mu, sigma) for mu in samples]

# 绘制后验预测分布
plt.hist(y_new_samples, bins=50, density=True, alpha=0.5, label="Posterior Predictive")
plt.hist(y, density=True, alpha=0.5, label="Observed Data")
plt.title("Posterior Predictive Distribution(MCMC)")
plt.legend()
plt.show()
