import gymnasium as gym

import random
import numpy as np
import torch

MAX_NUM_EPISODES = 100000
STEPS_PER_EPISODE = 300

env = gym.make("CartPole-v1", render_mode="rgb_array")


class SLP(torch.nn.Module):
    """
    A Single Layer Perceptron (SLP) class to approximate functions
    """

    def __init__(self, input_shape, output_shape, device=torch.device("cpu")):
        """
        :param input_shape: Shape/dimension of the input
        :param output_shape: Shape/dimension of the output
        :param device: The device (cpu or cuda) that the SLP should use to store the inputs for the forward pass
        """
        super(SLP, self).__init__()
        self.device = device
        self.input_shape = input_shape[0]
        self.hidden_shape = 40
        self.linear1 = torch.nn.Linear(self.input_shape, self.hidden_shape)
        self.out = torch.nn.Linear(self.hidden_shape, output_shape)

    def forward(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.out(x)
        return x


class LinearDecaySchedule(object):
    def __init__(self, initial_value, final_value, max_steps):
        assert initial_value > final_value, "initial_value should be > final_value"
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_factor = (initial_value - final_value) / max_steps

    def __call__(self, step_num):
        current_value = self.initial_value - self.decay_factor * step_num
        if current_value < self.final_value:
            current_value = self.final_value
        return current_value


class ShallowQLearner(object):
    def __init__(self, state_shape, action_shape, learning_rate=0.005, gamma=0.98):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.gamma = gamma
        self.learning_rate = learning_rate

        self.Q = SLP(state_shape, action_shape)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=1e-3)
        self.policy = self.epsilon_greedy_Q
        self.epsilon_max = 1.0
        self.epsilon_min = 0.005
        self.epsilon_decay = LinearDecaySchedule(initial_value=self.epsilon_max, final_value=self.epsilon_min,
                                                 max_steps=0.5 * MAX_NUM_EPISODES * STEPS_PER_EPISODE)
        self.step_num = 0

    def get_action(self, observation):
        return self.policy(observation)

    def epsilon_greedy_Q(self, observation):
        if random.random() < self.epsilon_decay(self.step_num):
            action = random.choice([a for a in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(observation).data.numpy())

        return action

    def learn(self, state, action, reward, next_state):
        td_target = reward + self.gamma * torch.max(self.Q(next_state))
        td_error = torch.nn.functional.mse_loss(self.Q(state)[action], td_target)
        self.Q_optimizer.zero_grad()
        td_error.backward()
        self.Q_optimizer.step()


if __name__ == "__main__":
    observation_shape = env.observation_space.shape
    action_shape = env.action_space.n
    agent = ShallowQLearner(observation_shape, action_shape)
    first_episode = True
    episode_rewards = list()

    for episode in range(MAX_NUM_EPISODES):
        obs, _ = env.reset()
        cum_reward = 0.0
        for step in range(STEPS_PER_EPISODE):
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
            cum_reward += reward

            if terminated or truncated:
                if first_episode:
                    max_reward = cum_reward
                    first_episode = False

                if cum_reward > max_reward:
                    max_reward = cum_reward
                print(
                    "\nEpisode#{} ended in {} steps. reward={} mean_reward={} best_reward={}".format(episode, step + 1,
                                                                                                     cum_reward,
                                                                                                     np.mean(
                                                                                                         episode_rewards),
                                                                                                     max_reward))
                break
