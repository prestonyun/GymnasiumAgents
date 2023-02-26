import gymnasium as gym
import numpy as np
import torch
import math
import random
from torch import Tensor, nn
import torch.nn.functional as F
import torch.nn.init as init
import matplotlib.pyplot as plt

losses_list = []
q_values_list = []

# Train the network
def train(env, agent, n_episodes, max_timesteps, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay):
    rewards = []
    epsilons = []
    for i in range(1, n_episodes+1):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * i / epsilon_decay)
        #epsilon = epsilon_start
        timestep = 0
        num_actions = 0

        for t in range(max_timesteps):
            # Select an action using an epsilon-greedy policy
            action = agent.act(obs, epsilon)

            # Take a step in the environment
            next_obs, reward, done, _, info = env.step(action)

            # Add the experience to the replay buffer
            agent.replay_buffer.add(obs, action, reward, next_obs, done)

            reward = reward * (1 + timestep / max_timesteps)

            # Update the Q-network using the replay buffer
            if agent.replay_buffer.size > batch_size:
                agent.update(batch_size, gamma)

            # Update the observation and total reward
            obs = next_obs
            total_reward += reward
            timestep += 1
            num_actions += 1

            if done:
                break

        rewards.append(total_reward)
        epsilons.append(epsilon)

        # Print the episode number, reward, and epsilon
        print(f"Episode {i}/{n_episodes}: reward = {total_reward}, epsilon = {epsilon:.2f}, 'buffer: , {agent.replay_buffer.size}")

    return rewards, epsilons

    
class QNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, n_actions=2):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_actions)

        init.uniform_(self.fc1.weight, -1.0 / math.sqrt(obs_dim), 1.0 / math.sqrt(obs_dim))
        init.uniform_(self.fc2.weight, -1.0 / math.sqrt(hidden_dim), 1.0 / math.sqrt(hidden_dim))
        init.uniform_(self.fc3.weight, 3e-3, 3e-3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class DQNAgent:
    def __init__(self, obs_dim, n_actions, hidden_dim, learning_rate, device):
        self.q_network = QNetwork(obs_dim, hidden_dim, n_actions).to(device)
        self.target_network = QNetwork(obs_dim,hidden_dim, n_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(50000)
        self.timestep = 0
        self.epsilon = 1.0
        self.device = device
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.target_network_update_freq = 1000

    def reset(self):
        #self.replay_buffer.clear()
        self.q_network.reset_parameters()
        self.target_network.load_state_dict(self.q_network.state_dict())

    def act(self, obs, epsilon):
        # Epsilon-greedy policy
        if np.random.random() < epsilon:
            action = np.random.randint(self.n_actions)
        else:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                q_values = self.q_network(obs_tensor)
                action = q_values.argmax().item()
                #print(obs_tensor, '--', q_values, '--', action)

        return action

    def update(self, batch_size, gamma):
        # Sample a batch of experiences from the replay buffer
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = self.replay_buffer.sample(batch_size)
        obs_batch = torch.FloatTensor(np.array(obs_batch)).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_obs_batch = torch.FloatTensor(np.array(next_obs_batch)).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)

        q_values = self.q_network(obs_batch)
        next_q_values = self.target_network(next_obs_batch)
        # Compute the target Q-values using the Bellman equation
        target_q_values = reward_batch + (1 - done_batch) * gamma * next_q_values.max(dim=1)[0]
        # Gather the Q-values for the actions that were taken
        q_values_for_actions = torch.gather(q_values, 0, action_batch.unsqueeze(1))
        #print(q_values_for_actions, '--', target_q_values)

        # Compute the loss using the smooth L1 loss function
        loss = F.smooth_l1_loss(q_values_for_actions, target_q_values.unsqueeze(1))
        losses_list.append(loss)
        q_values_list.append(q_values_for_actions)

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
        experience = (obs, action, reward, next_obs, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.idx] = experience
            self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        while len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*random.sample(self.buffer, batch_size))
        return obs_batch,action_batch, reward_batch,next_obs_batch,done_batch
    
    def clear(self):
        self.buffer = []
        self.idx = 0
        self.size = 0

# Initialize the environment and the agent
env = gym.make('CartPole-v1')
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
hidden_dim = 128
learning_rate = 5e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
agent = DQNAgent(obs_dim, n_actions, hidden_dim, learning_rate, device)

# Evaluate the learned policy for 100 episodes
num_episodes = 2500
total_reward = 0

rewards, epsilons = train(env, agent, num_episodes, 200, 32, 0.95, 0.99, 0.001, 1500)
env.close()

def plot_learning_curve(rewards, epsilons):
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.twinx()
    plt.plot(epsilons, color='r')
    plt.ylabel('Epsilon')
    plt.show()

# Plot the Q-values for each action over time
def plot_q_values(q_values_list):
    num_actions = q_values_list[0].shape[1]
    for action in range(num_actions):
        action_q_values = [Tensor.cpu(q_values[action].detach()).numpy() for q_values in q_values_list]
        plt.plot(action_q_values, label=f"Action {action}")
    plt.xlabel("Training episode")
    plt.ylabel("Q-value")
    plt.legend()
    plt.show()



plot_learning_curve(rewards, epsilons)
plot_q_values(q_values_list)