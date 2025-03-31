import gymnasium as gym
from gymnasium import envs
from ale_py import ALEInterface
ale = ALEInterface()

env_names = [spec for spec in envs.registry.keys()]
for name in sorted(env_names):
    print(name)

