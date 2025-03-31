import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, gamma
from scipy.special import psi  # 导入 psi 函数（即 digamma 函数）

# ==============================
# 生成合成数据（保持不变）
# ==============================
np.random.seed(42)
D = 2
N = 100
w_true = np.array([1.5, -0.8])
tau_true = 2.0
X = np.random.randn(N, D)
y = X @ w_true + np.random.normal(0, 1 / np.sqrt(tau_true), N)

# ==============================
# 先验参数（保持不变）
# ==============================
mu_0 = np.zeros(D)
Sigma_0 = np.eye(D) * 10
a_0 = 1.0
b_0 = 1.0

# ==============================
# 变分推断参数初始化（保持不变）
# ==============================
max_iter = 500
tol = 1e-4
mu_w = np.random.randn(D)
Sigma_w = np.eye(D)
E_tau = a_0 / b_0
elbo_history = []

# ==============================
# 变分推断主循环（仅修改 digamma 部分）
# ==============================
for iter in range(max_iter):
    # 更新 q(w)
    Sigma_w_inv = E_tau * X.T @ X + np.linalg.inv(Sigma_0)
    Sigma_w = np.linalg.inv(Sigma_w_inv)
    mu_w = Sigma_w @ (E_tau * X.T @ y + np.linalg.inv(Sigma_0) @ mu_0)

    # 更新 q(τ)
    residual = y - X @ mu_w
    expected_sq_error = np.sum(residual ** 2) + np.trace(X.T @ X @ Sigma_w)
    a_tau = a_0 + N / 2.0
    b_tau = b_0 + 0.5 * expected_sq_error
    E_tau_new = a_tau / b_tau

    # 计算 ELBO（替换 np.digamma 为 psi）
    log_p_w = -0.5 * (
            (mu_w - mu_0).T @ np.linalg.inv(Sigma_0) @ (mu_w - mu_0)
            + np.trace(np.linalg.inv(Sigma_0) @ Sigma_w)
    )

    # 修改点 1: 使用 psi 替代 np.digamma
    log_p_tau = (a_0 - 1) * (np.log(b_tau) - psi(a_tau)) - b_0 * E_tau_new

    log_likelihood = 0.5 * N * (np.log(E_tau_new) - np.log(2 * np.pi)) - 0.5 * E_tau_new * expected_sq_error

    entropy_w = 0.5 * np.log(np.linalg.det(Sigma_w)) + 0.5 * D * (1 + np.log(2 * np.pi))

    # 修改点 2: 使用 psi 替代 np.digamma
    entropy_tau = a_tau - np.log(b_tau) + np.log(gamma(a_tau).pdf(a_tau)) + (1 - a_tau) * psi(a_tau)

    elbo = log_p_w + log_p_tau + log_likelihood + entropy_w + entropy_tau
    elbo_history.append(elbo)

    if iter > 0 and np.abs(elbo - elbo_history[-2]) < tol:
        print(f"Converged at iteration {iter}")
        break

    E_tau = E_tau_new

# ==============================
# 结果可视化
# ==============================
plt.figure(figsize=(12, 4))

# ------------------------------
# 1. 权重后验分布
# ------------------------------
plt.subplot(131)
plt.scatter(w_true[0], w_true[1], c='red', s=100, label="真实权重", edgecolors='k')
plt.scatter(mu_w[0], mu_w[1], c='blue', s=100, label="变分均值", marker='x', linewidth=2)

# 绘制协方差椭圆
from matplotlib.patches import Ellipse

eigvals, eigvecs = np.linalg.eigh(Sigma_w)
angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
ellipse = Ellipse(mu_w, width=2 * np.sqrt(eigvals[0]), height=2 * np.sqrt(eigvals[1]),
                  angle=angle, edgecolor='blue', lw=2, facecolor='none')
plt.gca().add_patch(ellipse)
plt.xlabel("w1"), plt.ylabel("w2"), plt.title("权重后验分布")
plt.legend()

# ------------------------------
# 2. 预测结果
# ------------------------------
plt.subplot(132)
x_test = np.linspace(-3, 3, 100)
X_test = np.column_stack([x_test, 0.5 * x_test])  # 第二个特征与第一个相关

# 预测均值
y_pred_mean = X_test @ mu_w

# 预测方差 (考虑权重不确定性和噪声)
y_pred_var = np.diag(X_test @ Sigma_w @ X_test.T) + 1 / E_tau
y_pred_std = np.sqrt(y_pred_var)

plt.scatter(X[:, 0], y, alpha=0.6, label="训练数据")
plt.plot(x_test, y_pred_mean, 'r-', lw=2, label="预测均值")
plt.fill_between(x_test, y_pred_mean - 2 * y_pred_std, y_pred_mean + 2 * y_pred_std,
                 color='red', alpha=0.2, label="95%置信区间")
plt.xlabel("x1"), plt.ylabel("y"), plt.title("预测结果")
plt.legend()

# ------------------------------
# 3. ELBO 收敛曲线
# ------------------------------
plt.subplot(133)
plt.plot(elbo_history, 'o-', markersize=5)
plt.xlabel("迭代次数"), plt.ylabel("ELBO"), plt.title("ELBO收敛曲线")

plt.tight_layout()
plt.show()

# ==============================
# 输出结果
# ==============================
print(f"真实权重: {w_true}")
print(f"变分均值估计: {mu_w}")
print(f"变分协方差矩阵:\n{Sigma_w}")
print(f"噪声精度估计 E[τ]: {E_tau:.2f} (真实值 {tau_true})")