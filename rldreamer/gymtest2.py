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
    def __init__(self, obs_dim, n_actions, hidden_dim=128):
        super(DuelingQNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        # Define the Q-network architecture (e.g. using a hybrid transformer-RNN)
        self.transformer_rnn = TransformerRNN(obs_dim, hidden_dim=hidden_dim, num_heads=2, num_layers=2)
        self.fc_adv = nn.Linear(hidden_dim, n_actions)
        self.fc_val = nn.Linear(hidden_dim, 1)

        # Define the optimizer and loss function
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

    def forward(self, obs, action=None):
        # Compute the Q-values for the given state-action pair
        key_vectors, value_vectors = self.transformer_rnn(obs)
        adv = F.relu(self.fc_adv(value_vectors))
        val = self.fc_val(value_vectors)
        q_values = val + adv - adv.mean(dim=1, keepdim=True)

        if action is not None:
            q_values = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        return q_values

class VectorHologram:
    def __init__(self, dim):
        self.dim = dim
        self.data = np.zeros((dim,), dtype=np.complex128)

    def add(self, vector):
        assert len(vector) == self.dim
        self.data += np.fft.fft(vector)

    def dot(self, vector):
        assert len(vector) == self.dim
        result = np.fft.ifft(self.data * np.fft.fft(vector))
        return np.real(result)



class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = np.empty(capacity, dtype=object)
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

# Initialize the environment and the agent
env = gym.make('CartPole-v1')
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
key_dim = 128
value_dim = 128
hidden_dim = 1
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent = DQNAgent(key_dim, value_dim, n_actions, hidden_dim, learning_rate, device)
transformer_rnn = TransformerRNN(obs_dim, hidden_dim, hidden_dim, 1).to(device)

# Evaluate the learned policy for 100 episodes
num_episodes = 100
total_reward = 0

for episode in range(num_episodes):
    obs = env.reset()[0]
    done = False

    while not done:
        # Take the action with the highest Q-value
        with torch.no_grad():
            obs_tensor = torch.from_numpy(np.copy(obs)).unsqueeze(0).to(device)
            key_vectors, value_vectors = transformer_rnn(obs_tensor)
            print(key_vectors, value_vectors)
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