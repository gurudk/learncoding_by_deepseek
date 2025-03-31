import random

# 无放回采样
def simple_random_sample(data, k):
    return random.sample(data, k)

# 有放回采样
def random_sample_with_replacement(data, k):
    return [random.choice(data) for _ in range(k)]

# 示例
data = [1, 2, 3, 4, 5]
print(simple_random_sample(data, 2))        # 输出如 [3, 1]
print(random_sample_with_replacement(data, 2))  # 输出如 [5, 5]