import numpy as np

# 定义贝叶斯网络的概率参数
prob = {
    'S': [0.8, 0.2],  # P(S=0)=0.8, P(S=1)=0.2
    'L': {  # P(L | S)
        0: [0.99, 0.01],  # S=0时 P(L=0)=0.99, P(L=1)=0.01
        1: [0.9, 0.1]  # S=1时 P(L=0)=0.9,  P(L=1)=0.1
    },
    'C': {  # P(C | L)
        0: [0.95, 0.05],  # L=0时 P(C=0)=0.95, P(C=1)=0.05
        1: [0.1, 0.9]  # L=1时 P(C=0)=0.1,  P(C=1)=0.9
    }
}


# ================== 拒绝采样算法 ==================
def rejection_sampling(num_samples, evidence):
    samples = []
    valid_samples = 0

    for _ in range(num_samples):
        # 按拓扑顺序生成样本 (S -> L -> C)
        S = np.random.choice([0, 1], p=prob['S'])
        L = np.random.choice([0, 1], p=prob['L'][S])
        C = np.random.choice([0, 1], p=prob['C'][L])

        # 检查证据条件 (此处C=1)
        if C == evidence['C']:
            samples.append((S, L, C))
            valid_samples += 1

    # 计算后验概率 P(L=1 | C=1)
    if valid_samples == 0:
        return 0.0
    l1_count = sum(1 for s in samples if s[1] == 1)
    posterior = l1_count / valid_samples
    return posterior, valid_samples


# =============== 似然加权采样算法 ===============
def likelihood_weighting(num_samples, evidence):
    weights = []
    L_values = []

    for _ in range(num_samples):
        weight = 1.0

        # 生成非证据变量 (S, L)
        S = np.random.choice([0, 1], p=prob['S'])
        L = np.random.choice([0, 1], p=prob['L'][S])

        # 计算证据变量 (C=1) 的权重
        weight *= prob['C'][L][evidence['C']]

        weights.append(weight)
        L_values.append(L)

    # 计算加权后验概率
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    weighted_sum = sum(w * (1 if L == 1 else 0) for w, L in zip(weights, L_values))
    posterior = weighted_sum / total_weight
    return posterior, total_weight


# ================== 运行实验 ==================
if __name__ == "__main__":
    np.random.seed(42)  # 固定随机种子保证结果可复现

    # 使用拒绝采样估计 P(L=1 | C=1)
    num_samples = 100000
    post_rej, valid = rejection_sampling(num_samples, evidence={'C': 1})
    print(f"拒绝采样结果 (总样本={num_samples}, 有效样本={valid})")
    print(f"P(L=1 | C=1) ≈ {post_rej:.4f}\n")

    # 使用似然加权采样估计 P(L=1 | C=1)
    post_lw, total_w = likelihood_weighting(num_samples, evidence={'C': 1})
    print(f"似然加权采样结果 (总样本={num_samples}, 总权重={total_w:.2f})")
    print(f"P(L=1 | C=1) ≈ {post_lw:.4f}")