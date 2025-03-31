import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# 定义参数
shape_params = [1,2,3]  # 形状参数k的不同值
scale_param = [2,3,4]  # 尺度参数theta

# 设置x轴范围
x = np.linspace(gamma.ppf(0.01, shape_params[0], scale=scale_param),
                gamma.ppf(0.99, shape_params[-1], scale=scale_param), 100)

# 创建图形
plt.figure(figsize=(8, 6))

# 对于每个形状参数，绘制其对应的概率密度函数
for k in shape_params:
    for s in scale_param:
        y = gamma.pdf(x, a=k, scale=s)
        plt.plot(x, y, label=f'k={k}, θ={s}')

# 添加标题和标签
plt.title('Gamma Distribution PDF for Different Shape Parameters')
plt.xlabel('x')
plt.ylabel('PDF')
plt.legend()

# 显示图形
plt.show()