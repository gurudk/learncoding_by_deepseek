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
plt.scatter(x, y, alpha=0.6, label="观测数据")
plt.plot(x, w_true * x + b_true, 'r', label="真实直线")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# 转换为Pyro张量
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)


def model(x, y):
    # 定义先验分布
    w = pyro.sample("w", dist.Normal(0, 1))  # 先验 N(0,1)
    b = pyro.sample("b", dist.Normal(0, 1))  # 先验 N(0,1)
    sigma = pyro.sample("sigma", dist.HalfNormal(1))  # 半正态分布

    # 定义似然函数
    with pyro.plate("data", len(x)):
        mu = w * x + b
        pyro.sample("obs", dist.Normal(mu, sigma), obs=y)


def guide(x, y):
    # 定义变分参数
    w_loc = pyro.param("w_loc", torch.tensor(0.0))
    w_scale = pyro.param("w_scale", torch.tensor(1.0), constraint=dist.constraints.positive)
    b_loc = pyro.param("b_loc", torch.tensor(0.0))
    b_scale = pyro.param("b_scale", torch.tensor(1.0), constraint=dist.constraints.positive)
    sigma_loc = pyro.param("sigma_loc", torch.tensor(1.0), constraint=dist.constraints.positive)

    # 采样变分分布
    pyro.sample("w", dist.Normal(w_loc, w_scale))
    pyro.sample("b", dist.Normal(b_loc, b_scale))
    pyro.sample("sigma", dist.HalfNormal(sigma_loc))

# 初始化优化器和推断算法
optimizer = Adam({"lr": 0.01})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# 训练迭代
num_iterations = 2000
losses = []
for j in range(num_iterations):
    loss = svi.step(x_tensor, y_tensor)
    losses.append(loss)
    if j % 100 == 0:
        print(f"Iteration {j}, Loss = {loss:.2f}")

# 绘制损失曲线
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("ELBO Loss")
plt.show()

# 获取变分参数
w_post_mean = pyro.param("w_loc").item()
w_post_std = pyro.param("w_scale").item()
b_post_mean = pyro.param("b_loc").item()
b_post_std = pyro.param("b_scale").item()
sigma_post = pyro.param("sigma_loc").item()

print(f"变分推断结果:")
print(f"w = {w_post_mean:.2f} ± {w_post_std:.2f}")
print(f"b = {b_post_mean:.2f} ± {b_post_std:.2f}")
print(f"sigma = {sigma_post:.2f}")

import pymc as pm

with pm.Model() as mcmc_model:
    # 定义先验分布
    w = pm.Normal("w", mu=0, sigma=1)
    b = pm.Normal("b", mu=0, sigma=1)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # 定义似然函数
    mu = w * x + b
    likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=y)

    # 采样
    trace = pm.sample(2000, tune=1000, cores=1, return_inferencedata=True)


import arviz as az

# 后验分布可视化
az.plot_posterior(trace,var_names=["w", "b", "sigma"])
plt.show()

# 轨迹图
az.plot_trace(trace)
plt.show()