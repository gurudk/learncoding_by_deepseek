import numpy as np
import matplotlib.pyplot as plt

# 生成模拟数据
np.random.seed(42)
mu_true = 3.0
sigma = 1.0
n = 500
y = np.random.normal(mu_true, sigma, n)

# 先验参数
mu_0 = 0
tau_0 = 1.0

# 计算后验参数
y_bar = np.mean(y)
mu_n = (sigma**2 * mu_0 +tau_0**2 * n * y_bar)/(sigma**2 + n*tau_0**2)
tau_n = np.sqrt((sigma**2*tau_0**2)/(sigma**2 + tau_0**2 * n))

# 后验预测分布的均值和方差
pred_mu = mu_n
pred_var = tau_n**2 + sigma**2

# 绘制后验预测分布
x = np.linspace(pred_mu-4*np.sqrt(pred_var), pred_mu+4*np.sqrt(pred_var), 1000)
pdf = 1/np.sqrt(2*np.pi*pred_var)*np.exp(-0.5*(x-pred_mu)**2/pred_var)

plt.plot(x, pdf, label="Posterior Predictive")
plt.hist(y, density=True, alpha=0.5, label="Observed Data")
plt.title("Posterior Predictive Distribution(Anyalytic)")
plt.legend()
plt.show()



