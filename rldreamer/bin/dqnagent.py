import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from .nets.holographic_attn import DuelingQNetwork
from collections import deque
import random
import torch.nn.functional as F


class DQNAgent:
    def __init__(self, key_dim, value_dim, n_actions, hidden_dim, learning_rate, device):
        
        #self.q_network = HolographicAttentionNetwork(4, key_dim, value_dim, 2, n_actions).to(device)
        #self.target_network = HolographicAttentionNetwork(4, key_dim, value_dim, 2, n_actions).to(device)
        self.q_network = DuelingQNetwork(4, n_actions).to(device)
        self.target_network = DuelingQNetwork(4, n_actions).to(device)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(capacity=10000)
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
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = self.replay_buffer.sample(batch_size)
        obs_batch = torch.FloatTensor(obs_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(next_obs_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        # Compute the Q-values for the current and next observation batches
        q_values = self.q_network(obs_batch)
        q_values = q_values.gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_network(next_obs_batch)

        # Compute the target Q-values using the Bellman equation
        target_q_values = reward_batch + (1 - done_batch) * gamma * next_q_values.max(dim=1)[0]

        # Compute the loss between the Q-values and the target Q-values
        loss = F.smooth_l1_loss(q_values.gather(1, action_batch.unsqueeze(1)), target_q_values.unsqueeze(1))

        # Zero out the gradients and backpropagate the loss
        self.optimizer.zero_grad()
        loss.backward()

        # Clip the gradients to avoid exploding gradients
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Update the target network every target_network_update_freq timesteps
        if self.timestep % self.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Increment the timestep counter and decay the epsilon value
        self.timestep += 1
        self.epsilon = max(0.1, 1.0 - 0.9 * (self.timestep / 10000))


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = np.empty(capacity, dtype=np.object)
        self.idx = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.buffer[self.idx] = (obs, action, reward, next_obs, done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        batch = random.sample(self.buffer[:self.size], batch_size)
        obs, action, reward, next_obs, done = map(np.stack, zip(*batch))
        return obs, action, reward, next_obs, done

class DQN(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=128):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
