import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

# =========================
# 辅助函数：绘制协方差椭圆
# =========================
def plot_covariance_ellipse(mean, cov, nstd=2, **kwargs):
    """ 绘制协方差椭圆 """
    eigvals, eigvecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(eigvals)
    ellipse = plt.matplotlib.patches.Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        **kwargs
    )
    plt.gca().add_patch(ellipse)

# =========================
# 卡尔曼滤波类定义（多维版本）
# =========================
class KalmanFilterMultiDim:
    def __init__(self, initial_state, initial_covariance, F, H, Q, R):
        """
        参数初始化
        :param initial_state: 初始状态向量 (n维列向量)
        :param initial_covariance: 初始协方差矩阵 (n×n)
        :param F: 状态转移矩阵 (n×n)
        :param H: 观测矩阵 (m×n)
        :param Q: 过程噪声协方差矩阵 (n×n)
        :param R: 观测噪声协方差矩阵 (m×m)
        """
        self.x = initial_state  # 状态向量
        self.P = initial_covariance  # 协方差矩阵
        self.F = F  # 状态转移矩阵
        self.H = H  # 观测矩阵
        self.Q = Q  # 过程噪声
        self.R = R  # 观测噪声

        # 历史记录
        self.states = [self.x.copy()]
        self.covariances = [self.P.copy()]

    def predict(self):
        """ 预测阶段 """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """ 更新阶段 """
        # 计算卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        self.K = self.P @ self.H.T @ np.linalg.inv(S)

        # 更新状态和协方差
        self.x = self.x + self.K @ (z - self.H @ self.x)
        self.P = (np.eye(len(self.x)) - self.K @ self.H) @ self.P

        # 记录历史
        self.states.append(self.x.copy())
        self.covariances.append(self.P.copy())


# =========================
# 参数设置
# =========================
np.random.seed(42)

# 时间参数
dt = 1.0  # 时间步长 (秒)
num_steps = 50  # 总时间步数

# 状态向量定义: [x, y, vx, vy]
INITIAL_STATE = np.array([0.0, 0.0, 1.0, 0.5])  # 初始位置(0,0), 速度(1,0.5)

# 状态转移矩阵 (匀速模型)
F = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# 观测矩阵 (仅观测位置)
H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])

# 过程噪声协方差 (加速扰动)
Q_scale = 0.1
Q = np.array([
    [dt ** 4 / 4 * Q_scale, 0, dt ** 3 / 2 * Q_scale, 0],
    [0, dt ** 4 / 4 * Q_scale, 0, dt ** 3 / 2 * Q_scale],
    [dt ** 3 / 2 * Q_scale, 0, dt ** 2 * Q_scale, 0],
    [0, dt ** 3 / 2 * Q_scale, 0, dt ** 2 * Q_scale]
])

# 观测噪声协方差
R_scale = 1.0
R = R_scale * np.eye(2)

# 初始协方差
INITIAL_COV = np.diag([5.0, 5.0, 2.0, 2.0])  # 初始位置和速度的不确定性

# =========================
# 生成模拟数据
# =========================
# 真实轨迹
true_states = [INITIAL_STATE.copy()]
for _ in range(num_steps):
    # 添加过程噪声 (加速度扰动)
    noise = np.random.multivariate_normal(mean=[0, 0, 0, 0], cov=Q)
    new_state = F @ true_states[-1] + noise
    true_states.append(new_state)

# 生成带噪声的观测值 (仅位置)
measurements = []
for state in true_states[1:]:
    pos_noise = np.random.multivariate_normal(mean=[0, 0], cov=R)
    measurements.append(H @ state + pos_noise)

# =========================
# 执行卡尔曼滤波
# =========================
kf = KalmanFilterMultiDim(
    initial_state=INITIAL_STATE,
    initial_covariance=INITIAL_COV,
    F=F,
    H=H,
    Q=Q,
    R=R
)

# 迭代处理每个观测
for z in measurements:
    kf.predict()
    kf.update(z)

# =========================
# 可视化结果
# =========================
plt.figure(figsize=(12, 8))

# 提取数据
true_x = [s[0] for s in true_states]
true_y = [s[1] for s in true_states]
meas_x = [z[0] for z in measurements]
meas_y = [z[1] for z in measurements]
est_x = [s[0] for s in kf.states]
est_y = [s[1] for s in kf.states]

# 轨迹对比
plt.subplot(2, 2, (1, 2))
plt.plot(true_x, true_y, 'g-', label='True Trajectory')
plt.plot(meas_x, meas_y, 'r.', markersize=8, alpha=0.5, label='Measurements')
plt.plot(est_x, est_y, 'b--', linewidth=2, label='Kalman Estimate')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('2D Trajectory Tracking')
plt.legend()

# 协方差椭圆可视化（最后一步）
plt.subplot(2, 2, 3)
plot_covariance_ellipse(
    mean=[est_x[-1], est_y[-1]],
    cov=kf.covariances[-1][:2, :2],
    nstd=2,
    color='blue',
    alpha=0.3
)
plt.scatter(true_x[-1], true_y[-1], c='green', marker='*', s=100, label='True Position')
plt.scatter(est_x[-1], est_y[-1], c='blue', s=50, label='Estimate')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Final Estimation Uncertainty (2σ Ellipse)')
plt.legend()

# 速度收敛过程
plt.subplot(2, 2, 4)
true_vx = [s[2] for s in true_states]
true_vy = [s[3] for s in true_states]
est_vx = [s[2] for s in kf.states]
est_vy = [s[3] for s in kf.states]
plt.plot(true_vx, 'g-', label='True Vx')
plt.plot(true_vy, 'g--', label='True Vy')
plt.plot(est_vx, 'b-', label='Estimated Vx')
plt.plot(est_vy, 'b--', label='Estimated Vy')
plt.xlabel('Time Step')
plt.ylabel('Velocity')
plt.title('Velocity Estimation')
plt.legend()

plt.tight_layout()
plt.show()


