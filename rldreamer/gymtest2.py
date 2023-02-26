import gymnasium as gym
import numpy as np
import torch
import math
import random
from torch import nn
from torch import optim
import torch.nn.functional as F

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
        gru_embeddings, hidden = self.gru(obs.view(obs.shape[0], obs.shape[1], -1))

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
        obs, _ = env.reset()
        done = False
        total_reward = 0
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * i / epsilon_decay)

        for t in range(max_timesteps):
            # Select an action using an epsilon-greedy policy
            action = agent.act(obs, epsilon)

            # Take a step in the environment
            next_obs, reward, done, _, info = env.step(action)

            # Add the experience to the replay buffer
            agent.replay_buffer.add(obs, action, reward, next_obs, done)

            # Update the Q-network using the replay buffer
            if agent.replay_buffer.size > batch_size:
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

    
class DuelingQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, key_dim, value_dim, hidden_dim):
        super(DuelingQNetwork, self).__init__()

        self.fc1 = nn.Linear(num_inputs, hidden_dim)
        self.fc_key = nn.Linear(hidden_dim, key_dim)
        self.fc_val = nn.Linear(hidden_dim, value_dim)
        self.fc_adv = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        key = F.relu(self.fc_key(x))
        val = F.relu(self.fc_val(x))
        adv = F.relu(self.fc_adv(x))

        val = val.view(val.size(0), 1)
        adv = adv.view(adv.size(0), -1)

        q_values = val + adv - adv.mean(1, keepdim=True)
        return q_values




    
class DQNAgent:
    def __init__(self, obs_dim, key_dim, value_dim, n_actions, hidden_dim, learning_rate, device):
        self.q_network = DuelingQNetwork(obs_dim, n_actions, key_dim, value_dim, hidden_dim).to(device)
        self.target_network = DuelingQNetwork(obs_dim, n_actions, key_dim, value_dim, hidden_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(10000)
        self.timestep = 0
        self.epsilon = 1.0
        self.device = device
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.target_network_update_freq = 1000

        self.key_transform = nn.Sequential(
            nn.Linear(4, key_dim),
            nn.ReLU(),
            nn.Linear(key_dim, key_dim)
        ).to(device)
        self.value_transform = nn.Sequential(
            nn.Linear(4, value_dim),
            nn.ReLU(),
            nn.Linear(value_dim, value_dim)
        ).to(device)


    def act(self, obs, epsilon):
        # Epsilon-greedy policy
        if np.random.random() < epsilon:
            action = np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                key_vectors = self.key_transform(obs_tensor).to(self.device)
                value_vectors = self.value_transform(obs_tensor).to(self.device)
                q_values = self.q_network(obs_tensor)
                action = q_values.argmax().item()

        return action

    def update(self, batch_size, gamma):
        # Sample a batch of experiences from the replay buffer
        obs_key_batch, obs_value_batch, action_batch, reward_batch, next_obs_key_batch, next_obs_value_batch, done_batch = self.replay_buffer.sample(batch_size)
        obs_key_batch = torch.FloatTensor(obs_key_batch).to(self.device)
        obs_value_batch = torch.FloatTensor(obs_value_batch).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_obs_key_batch = torch.FloatTensor(next_obs_key_batch).to(self.device)
        next_obs_value_batch = torch.FloatTensor(next_obs_value_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        # Compute the Q-values for the current and next observation batches
        obs_combined = torch.cat((obs_key_batch, obs_value_batch), dim=-1).to(self.device)

        q_values = self.q_network(obs_combined)
        #q_values = q_values.gather(1, action_batch.unsqueeze(1))

        next_obs_combined = torch.cat((next_obs_key_batch, next_obs_value_batch), dim=-1).to(self.device)
        next_q_values = self.target_network(next_obs_combined)
        # Compute the target Q-values using the Bellman equation
        target_q_values = reward_batch + (1 - done_batch) * gamma * next_q_values.max(dim=0)[0]
        # Gather the Q-values for the actions that were taken
        q_values_for_actions = q_values.gather(0, action_batch.unsqueeze(1).long())


        # Compute the loss using the smooth L1 loss function
        loss = F.smooth_l1_loss(q_values_for_actions, target_q_values.unsqueeze(1))


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
        self.buffer = []
        self.idx = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        obs_key = obs[0]  # extract key from observation
        obs_value = obs[1]  # extract value from observation
        next_obs_key = next_obs[0]  # extract key from next observation
        next_obs_value = next_obs[1]  # extract value from next observation
        experience = (obs_key, obs_value, action, reward, next_obs_key, next_obs_value, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.idx] = experience
            self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        while len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        obs_key_batch, obs_value_batch, action_batch, reward_batch, next_obs_key_batch, next_obs_value_batch, done_batch = zip(*random.sample(self.buffer, batch_size))
        return obs_key_batch, obs_value_batch, action_batch, reward_batch, next_obs_key_batch, next_obs_value_batch, done_batch

# Initialize the environment and the agent
env = gym.make('CartPole-v1')
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
key_dim = 4
value_dim = 1
hidden_dim = 128
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
agent = DQNAgent(obs_dim, key_dim, value_dim, n_actions, hidden_dim, learning_rate, device)
transformer_rnn = TransformerRNN(obs_dim, hidden_dim, hidden_dim, 1).to(device)

# Evaluate the learned policy for 100 episodes
num_episodes = 10000
total_reward = 0

train(env, agent, num_episodes, 200, 2, 0.99, 1, 0.1, 1000)
env.close()

""" for episode in range(num_episodes):
    obs = env.reset()[0]
    done = False

    while not done:
        # Take the action with the highest Q-value
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            key_vectors = agent.key_transform(obs_tensor).to(device)
            value_vectors = agent.value_transform(obs_tensor).to(device)
            q_values = agent.q_network(key_vectors, value_vectors)
            action = q_values.argmax().item()


        # Step the environment and accumulate the reward
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

    # Print the total reward for the episode
    print(f"Episode {episode}, total reward = {total_reward}, epsilon = {agent.epsilon:.2f}, reward = {reward}")
    total_reward = 0

# Close the environment
env.close() """