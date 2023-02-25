import gymnasium as gym
import numpy as np
import torch
import math
from torch import nn
from torch import optim
from collections import deque
from rldreamer.bin.nets import holographic_attn

# Define the environment
env = gym.make('CartPole-v1')
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

# Define the reinforcement learning algorithm
class DQN(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=128):
        super(DQN, self).__init__()

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        # Define the Q-network architecture (e.g. using a hybrid transformer-RNN)
        self.transformer_rnn = TransformerRNN(obs_dim=obs_dim, key_dim=64, hidden_dim=hidden_dim, n_layers=2)
        self.holographic_nn = holographic_attn.HolographicAttentionNetwork(key_dim=64, value_dim=hidden_dim, num_slots=256)

        self.fc = nn.Linear(hidden_dim, n_actions)

        # Define the optimizer and loss function
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

    def forward(self, obs, action=None):
        # Compute the Q-values for the given state-action pair
        key_vectors, value_vectors = self.transformer_rnn(obs)
        q_values = []
        for i in range(self.n_actions):
            q_value = self.holographic_nn(key_vectors, value_vectors, i)
            q_values.append(q_value)
        q_values = torch.stack(q_values, dim=1)
        if action is not None:
            q_values = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        return q_values

    def update(self, replay_buffer, batch_size, gamma):
        # Sample a batch of experiences from the replay buffer
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = replay_buffer.sample(batch_size)
        obs_batch = torch.FloatTensor(obs_batch)
        action_batch = torch.LongTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        next_obs_batch = torch.FloatTensor(next_obs_batch)
        done_batch = torch.FloatTensor(done_batch)

        # Compute the Q-value targets using the Bellman equation and target network
        with torch.no_grad():
            target_q_values = reward_batch + (1 - done_batch) * gamma * torch.max(self.forward(next_obs_batch), dim=1)[0]

        # Compute the Q-values for the given state-action pairs
        q_values = self.forward(obs_batch, action_batch)

        # Compute the loss between the predicted and target Q-values
        loss = self.loss_fn(q_values, target_q_values)

        # Zero the gradients and backpropagate the loss
        self.optimizer.zero_grad()
        loss.backward()

        # Clip the gradients to avoid exploding gradients
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # Update the parameters using the optimizer
        self.optimizer.step()


# Define the observation function
class TransformerRNN(nn.Module):
    def __init__(self, obs_dim, key_dim, hidden_dim, n_layers):
        super(TransformerRNN, self).__init__()

        self.obs_dim = obs_dim
        self.key_dim = key_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Define the self-attention layer
        self.self_attn = nn.MultiheadAttention(embed_dim=obs_dim, kdim=key_dim, vdim=hidden_dim, num_heads=4)

        # Define the RNN layers
        self.rnn = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)

    def forward(self, obs):
        # Compute the key vectors using the self-attention layer
        key_vectors, _ = self.self_attn(obs, obs, obs)

        # Pass the key vectors through the RNN layers to capture the temporal dynamics
        _, (h, c) = self.rnn(key_vectors)

        # Return the final hidden state of the RNN layers as the output
        return h[-1]


# Train the network
def train(env, agent, n_episodes, max_timesteps, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay):
    rewards = []
    epsilons = []
    for i in range(1, n_episodes+1):
        obs = env.reset()
        done = False
        total_reward = 0
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * i / epsilon_decay)

        for t in range(max_timesteps):
            # Select an action using an epsilon-greedy policy
            action = agent.act(obs, epsilon)

            # Take a step in the environment
            next_obs, reward, done, info = env.step(action)

            # Add the experience to the replay buffer
            agent.replay_buffer.add(obs, action, reward, next_obs, done)

            # Update the Q-network using the replay buffer
            if agent.replay_buffer.size() > batch_size:
                agent.update(batch_size, gamma)

            # Update the observation and total reward
            obs = next_obs
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)
        epsilons.append(epsilon)

        # Print the episode number, reward, and epsilon
        print(f"Episode {i}/{n_episodes}: reward = {total_reward}, epsilon = {epsilon:.2f}")

    return rewards, epsilons

    
# Generate policies
def generate_policy(env, agent, transformer_rnn, h_anet, device):
    obs = env.reset()
    done = False

    while not done:
        # Convert the observation to a PyTorch tensor and add a batch dimension
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

        # Pass the observation through the transformer-RNN to generate the key and value vectors
        key_vectors, value_vectors = transformer_rnn(obs_tensor)

        # Pass the key and value vectors through the holographic attention network to compute the Q-values
        q_values = h_anet(key_vectors, value_vectors)

        # Select the action with the highest Q-value
        action = agent.get_action(q_values)

        # Take a step in the environment
        obs, _, done, _ = env.step(action)

    return

