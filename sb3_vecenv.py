from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

# 创建 VecEnv（4个并行环境）
env = make_vec_env(
    "CartPole-v1",
    n_envs=4,
    vec_env_cls=DummyVecEnv,  # 多进程并行
    env_kwargs={"render_mode": "rgb_array"},
)

# 初始化 PPO 算法
model = PPO("MlpPolicy", env, verbose=1)

# 训练
model.learn(total_timesteps=100000)

# 测试
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, rewards, dones, _ = env.step(action)
    env.render()  # 渲染所有环境（需支持多窗口）

env.close()