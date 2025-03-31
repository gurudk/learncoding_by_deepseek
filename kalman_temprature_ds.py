import numpy as np
import matplotlib.pyplot as plt


# =========================
# 卡尔曼滤波类定义
# =========================
class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        """
        参数初始化
        :param initial_state: 初始状态估计 (标量)
        :param initial_covariance: 初始协方差 (标量)
        :param process_noise: 过程噪声方差 Q (标量)
        :param measurement_noise: 观测噪声方差 R (标量)
        """
        # 状态转移矩阵 (标量版)
        self.F = 1  # 恒温模型

        # 观测矩阵 (标量版)
        self.H = 1

        # 噪声参数
        self.Q = process_noise
        self.R = measurement_noise

        # 初始状态
        self.x = initial_state
        self.P = initial_covariance

        # 历史记录
        self.states = [self.x]
        self.covariances = [self.P]
        self.kalman_gains = []

    def predict(self):
        """ 预测阶段 """
        self.x = self.F * self.x  # 状态预测
        self.P = self.F * self.P * self.F + self.Q  # 协方差预测

    def update(self, z):
        """ 更新阶段 """
        # 计算卡尔曼增益
        S = self.H * self.P * self.H + self.R
        self.K = (self.P * self.H) / S  # 标量除法

        # 更新状态估计
        self.x = self.x + self.K * (z - self.H * self.x)

        # 更新协方差
        self.P = (1 - self.K * self.H) * self.P

        # 记录数据
        self.states.append(self.x)
        self.covariances.append(self.P)
        self.kalman_gains.append(self.K)


# =========================
# 参数设置
# =========================
np.random.seed(42)  # 固定随机种子

# 真实参数
TRUE_TEMP = 25.0  # 真实温度 (°C)
NUM_STEPS = 50  # 时间步数

# 噪声参数
PROCESS_NOISE_VAR = 0.01  # 过程噪声方差 Q
MEASUREMENT_NOISE_VAR = 0.1  # 观测噪声方差 R

# 初始化参数
INITIAL_GUESS = 52.0  # 初始猜测温度
INITIAL_COVARIANCE = 1.0  # 初始协方差

# =========================
# 生成模拟数据
# =========================
# 生成真实温度（含过程噪声）
true_temps = [TRUE_TEMP]
for _ in range(NUM_STEPS):
    # 过程噪声：真实温度变化
    true_temp = true_temps[-1] + np.random.normal(0, np.sqrt(PROCESS_NOISE_VAR))
    true_temps.append(true_temp)

# 生成带噪声的观测值
measurements = [t + np.random.normal(0, np.sqrt(MEASUREMENT_NOISE_VAR)) for t in true_temps[1:]]

# =========================
# 执行卡尔曼滤波
# =========================
kf = KalmanFilter(
    initial_state=INITIAL_GUESS,
    initial_covariance=INITIAL_COVARIANCE,
    process_noise=PROCESS_NOISE_VAR,
    measurement_noise=MEASUREMENT_NOISE_VAR
)

for z in measurements:
    kf.predict()
    kf.update(z)

# =========================
# 可视化结果
# =========================
plt.figure(figsize=(12, 8))

# 温度对比图
plt.subplot(3, 1, 1)
plt.plot(true_temps, 'g-', label='True Temperature')
plt.plot(measurements, 'r.', markersize=8, label='Measurements')
plt.plot(kf.states, 'b-', label='Kalman Estimate')
plt.axhline(TRUE_TEMP, color='gray', linestyle='--', alpha=0.5)
plt.ylabel('Temperature (°C)')
plt.title('Kalman Filter Temperature Estimation')
plt.legend()

# 协方差变化图
plt.subplot(3, 1, 2)
plt.plot(kf.covariances, 'm-')
plt.ylabel('Covariance')
plt.title('Estimation Uncertainty')

# 卡尔曼增益变化图
plt.subplot(3, 1, 3)
plt.plot(kf.kalman_gains, 'c-')
plt.ylabel('Kalman Gain')
plt.xlabel('Time Step')

plt.tight_layout()
plt.show()