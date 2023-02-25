import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from .nets.holographic_attn import HolographicAttentionNetwork
from collections import deque
import random
import torch.nn.functional as F


class DQNAgent:
    def __init__(self, key_dim, value_dim, n_actions, hidden_dim, learning_rate, device):
        self.q_network = HolographicAttentionNetwork(4, key_dim, value_dim, 2, n_actions).to(device)
        self.target_network = HolographicAttentionNetwork(4, key_dim, value_dim, 2, n_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = deque(maxlen=10000)
        self.replay_buffer_capacity = 10000
        self.timestep = 0
        self.epsilon = 1.0
        self.device = device
        self.learning_rate = learning_rate

    def act(self, obs, epsilon):
        # Epsilon-greedy policy
        if np.random.random() < epsilon:
            action = np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                q_values = self.q_network(obs_tensor)
                action = q_values.argmax().item()

        return action

    def update(self, batch_size, gamma):
        # Sample a batch of experiences from the replay buffer
        batch = np.random.choice(self.replay_buffer, batch_size, replace=False)

        # Unpack the batch into separate arrays for each element of the experience tuple
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*batch)
        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        # Compute the Q-values for the current and next observation batches
        q_values = self.q_network(obs_batch)
        next_q_values = self.target_network(next_obs_batch)

        # Compute the target Q-values using the Bellman equation
        target_q_values = reward_batch + (1 - done_batch) * gamma * next_q_values.max(dim=1)[0]

        # Compute the loss between the Q-values and the target Q-values
        loss = self.loss_function(q_values.gather(1, action_batch.unsqueeze(1)), target_q_values.unsqueeze(1))

        # Zero out the gradients and backpropagate the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target network every target_network_update_freq timesteps
        if self.timestep % self.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Increment the timestep counter and decay the epsilon value
        self.timestep += 1
        self.epsilon = max(0.1, 1.0 - 0.9 * (self.timestep / 10000))
