#!/usr/bin/env python
# Handy script for exploring Gym environment's spaces | Praveen Palanisamy
# Chapter 4, Hands-on Intelligent Agents with OpenAI Gym, 2018

import sys
import gymnasium as gym
from gymnasium.spaces import *
from ale_py import ALEInterface

ale = ALEInterface()

def print_spaces(space):
    print(space)
    if isinstance(space, Box):  # Print lower and upper bound if it's a Box space
        print("\n space.low: ", space.low)
        print("\n space.high: ", space.high)


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    print("Observation Space:")
    print_spaces(env.observation_space)
    print("Action Space:")
    print_spaces(env.action_space)
    print(env)
    try:
        print("Action description/meaning:", env.unwrapped.get_action_meanings())
    except AttributeError:
        pass
