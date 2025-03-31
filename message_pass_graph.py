import numpy as np


# 定义变量和因子
class Variable:
    def __init__(self, name):
        self.name = name
        self.neighbors = []  # 相邻的因子节点

    def add_neighbor(self, factor):
        self.neighbors.append(factor)


class Factor:
    def __init__(self, name, variables, potential):
        self.name = name
        self.variables = variables  # 连接的变量节点
        self.potential = potential  # 势函数（条件概率表）


# 初始化变量和因子
X1 = Variable("X1")
X2 = Variable("X2")
X3 = Variable("X3")

# 定义因子（转移概率）
f1 = Factor("f1", [X1, X2], {
    (0, 0): 0.7, (0, 1): 0.3,
    (1, 0): 0.2, (1, 1): 0.8
})

f2 = Factor("f2", [X2, X3], {
    (0, 0): 0.7, (0, 1): 0.3,
    (1, 0): 0.2, (1, 1): 0.8
})

# 连接变量和因子
X1.add_neighbor(f1)
X2.add_neighbor(f1)
X2.add_neighbor(f2)
X3.add_neighbor(f2)

# 初始化消息
messages = {
    (X1, f1): np.array([0.6, 0.4]),  # X1的初始概率
    (X2, f1): np.array([1.0, 1.0]),  # 初始消息为均匀分布
    (X2, f2): np.array([1.0, 1.0]),
    (X3, f2): np.array([1.0, 1.0]),
}


# 前向传递（从X1到X3）
def send_message(from_var, to_factor):
    incoming_messages = []
    for factor in from_var.neighbors:
        if factor != to_factor:
            msg = messages.get((from_var, factor), np.array([1.0, 1.0]))
            incoming_messages.append(msg)
    combined = np.prod(incoming_messages, axis=0)
    messages[(from_var, to_factor)] = combined


# 反向传递（从X3到X1）
def send_factor_message(from_factor, to_var):
    other_vars = [var for var in from_factor.variables if var != to_var]
    other_var = other_vars[0]
    msg = messages.get((other_var, from_factor), np.array([1.0, 1.0]))

    potential = from_factor.potential
    summed_msg = np.zeros(2)
    for state in [0, 1]:
        total = 0.0
        for other_state in [0, 1]:
            key = (other_state, state) if other_var.name == "X1" else (state, other_state)
            total += potential[key] * msg[other_state]
        summed_msg[state] = total
    messages[(from_factor, to_var)] = summed_msg


# 执行消息传递
# 前向传递
send_message(X1, f1)
send_factor_message(f1, X2)
send_factor_message(f2, X3)

# 反向传递
send_factor_message(f2, X2)
send_factor_message(f1, X2)

# 计算X2的边缘概率
belief_X2 = np.prod([
    messages.get((f1, X2), np.array([1.0, 1.0])),
    messages.get((f2, X2), np.array([1.0, 1.0]))
], axis=0)

# 归一化
belief_X2 /= np.sum(belief_X2)

print(f"P(X2=0) = {belief_X2[0]:.4f}, P(X2=1) = {belief_X2[1]:.4f}")