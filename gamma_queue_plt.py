from scipy.stats import gamma
import matplotlib.pyplot as plt
import numpy as np


k = 2  # 形状参数
theta = 1/30  # 尺度参数

x = np.linspace(0, 1, 100)  # 时间范围从0到1小时
y = gamma.pdf(x, a=k, scale=theta)
plt.plot(x, y, label='Erlang (Gamma) Distribution')
plt.xlabel('Time (hours)')
plt.ylabel('PDF')
plt.title('Probability Density Function of Service Time')
plt.legend()
plt.show()