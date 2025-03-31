import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import matplotlib.pyplot as plt

# 真实参数
w_true = 2.0
b_true = 1.0
sigma_true = 0.5

# 生成数据
N = 100
x = np.linspace(0, 1, N)
y = w_true * x + b_true + np.random.normal(0, sigma_true, N)

# 可视化
# plt.scatter(x, y, alpha=0.6, label="观测数据")
# plt.plot(x, w_true * x + b_true, 'r', label="真实直线")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()

import pymc as pm

import arviz as az

with pm.Model() as mcmc_model:
    # 定义先验分布
    w = pm.Normal("w", mu=0, sigma=1)
    b = pm.Normal("b", mu=0, sigma=1)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # 定义似然函数
    mu = w * x + b
    likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

    # 采样
    trace = pm.sample(2000, nuts_sampler="nutpie",tune=1000, cores=2, return_inferencedata=False)
    print(trace)

    # 后验分布可视化
    az.plot_trace(trace, combined=True)
    plt.show()

    # 轨迹图
    az.plot_trace(trace)
    plt.show()