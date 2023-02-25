import gymnasium as gym
import numpy as np
import torch
import math
from torch import nn
from torch import optim
from bin.nets import holographic_attn
from bin import dqnagent

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.cuda.set_per_process_memory_fraction(0.5, device=0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, dropout_prob=0.2):
        super(TransformerRNN, self).__init__()
        
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout_prob)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.ReLU(), nn.Linear(hidden_dim * 4, hidden_dim))
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
    def forward(self, obs):
        if obs.ndim == 2:
            # If the input tensor has only two dimensions, add a batch dimension
            obs = obs.unsqueeze(0)

        # Compute the GRU embeddings of the input sequence
        gru_embeddings, hidden = self.gru(obs)

        # Compute the self-attention vectors by passing the GRU embeddings through the self-attention layer
        key_vectors, _ = self.self_attn(gru_embeddings, gru_embeddings, gru_embeddings)

        # Pass the key vectors through a feedforward network to obtain the value vectors
        value_vectors = self.ffn(key_vectors)

        return key_vectors, value_vectors



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

# Initialize the environment and the agent
env = gym.make('CartPole-v1')
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
key_dim = 128
value_dim = 128
hidden_dim = 8
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent = dqnagent.DQNAgent(key_dim, value_dim, n_actions, hidden_dim, learning_rate, device)
transformer_rnn = TransformerRNN(obs_dim, hidden_dim, hidden_dim, 1).to(device)

# Evaluate the learned policy for 100 episodes
num_episodes = 1000
total_reward = 0

for episode in range(num_episodes):
    obs = env.reset()[0]
    done = False

    while not done:
        # Take the action with the highest Q-value
        with torch.no_grad():
            obs_tensor = torch.from_numpy(np.copy(obs)).unsqueeze(0).to(device)
            key_vectors, value_vectors = transformer_rnn(obs_tensor)
            
            q_values = agent.q_network(key_vectors, value_vectors)
            action = q_values.argmax().item()

        # Step the environment and accumulate the reward
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

    # Print the total reward for the episode
    print(f"Episode {episode}: total reward = {total_reward}")
    total_reward = 0

# Close the environment
env.close()
