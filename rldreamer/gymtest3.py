import random
import torch

import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import Tensor

q_values_list = []

class DQNAgent():
    def __init__(self, state_size, action_size, hidden_size, lr, gamma, capacity, batch_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        ).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.batch_size = batch_size
        self.gamma = gamma
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = []
        self.capacity = capacity
        self.position = 0

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.q_net[-1].out_features -1)
        else:
            state = torch.FloatTensor(state).to(self.device)
            q_values = self.q_net(state)
            action = q_values.max(-1)[1].item()
            q_values_list.append(q_values)
            return action

    def optimize(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)
        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        q_values = self.q_net(state).gather(1, action)

        next_q_values = self.q_net(next_state).detach().max(1)[0].unsqueeze(1)
        expected_q_values = reward + self.gamma * next_q_values * (1 - done)

        loss = self.loss_fn(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def remember(self, state, action, reward, next_state, done):
        if len(self.replay_buffer) < self.capacity:
            self.replay_buffer.append(None)
        while len(self.replay_buffer) > self.capacity:
            self.replay_buffer.pop(0)
        self.replay_buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def learn(self, num_episodes, max_steps, epsilon_start, epsilon_end, epsilon_decay, env):
        epsilon = epsilon_start
        rewards, epsilons = [], []
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            for step in range(max_steps):
                #env.render()
                action = self.act(state, epsilon)
                next_state, reward, done, _, info = env.step(action)
                total_reward += reward
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.optimize()
                if done:
                    break
            rewards.append(total_reward)
            epsilons.append(epsilon)
            epsilon = max(epsilon_end, epsilon_decay * epsilon)
            print(f"Episode {episode}/{num_episodes}: reward = {total_reward}, epsilon = {epsilon:.2f}")
        return rewards, epsilons
    
class PrioritizedDQNAgent():
    def __init__(self, state_size, action_size, hidden_size, lr, gamma, capacity, batch_size, alpha, beta_start, beta_annealing_steps):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        ).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.batch_size = batch_size
        self.gamma = gamma
        self.loss_fn = nn.MSELoss()
        self.capacity = capacity
        self.replay_buffer = [(None, None, None, None, None, None)] * capacity  # initialize with empty tuples
        self.priority_pos = 0
        self.priorities = np.zeros(capacity)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_annealing_steps = beta_annealing_steps
        self.steps = 0

    def remember(self, state, action, reward, next_state, done):
        # Compute the priority of the new transition based on its TD error
        state_tensor = Tensor(state).to(self.device)
        next_state_tensor = Tensor(next_state).to(self.device)
        q_value = self.q_net(state_tensor)[action]
        target_q_value = reward + self.gamma * self.q_net(next_state_tensor).max()
        td_error = abs(q_value - target_q_value).cpu().detach().numpy()
        priority = pow(td_error + 1e-6, self.alpha)
        if len(self.replay_buffer) < self.capacity:
            self.replay_buffer.append(None)
        while len(self.replay_buffer) > self.capacity:
            self.replay_buffer.pop(0)
        self.priorities[self.priority_pos] = priority
        self.replay_buffer[self.priority_pos] = (state, action, reward, next_state, done)
        self.priority_pos = (self.priority_pos + 1) % self.capacity

    def sample(self, beta=None):
        # Compute the probabilities for each transition based on their priorities
        if beta is None:
            beta = self.beta
        probs = self.priorities / self.priorities.sum()
        # Sample transitions from the replay buffer based on their probabilities
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, p=probs)
        states, actions, rewards, next_states, dones = zip(*[self.replay_buffer[i] for i in indices])
        # Compute the importance sampling weights for the transitions
        weights = ((1 / (len(self.replay_buffer) * probs[indices])) ** beta).astype(np.float32)
        weights /= weights.max()
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones), np.array(weights), indices

    def optimize(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        # Anneal the beta value over time
        beta = min(1, self.beta + (1 - self.beta) * self.steps / self.beta_annealing_steps)
        # Sample transitions from the replay buffer based on their priorities and compute their weights
        states, actions, rewards, next_states, dones, weights, indices = self.sample(beta)
        # Convert the data to tensors and move them to the GPU
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        # Compute the Q-values for the current state-action pairs
        q_values = self.q_net(states).gather(1, actions)
        # Compute the TD targets for the next states
        next_q_values = self.q_net(next_states).detach().max(1)[0].unsqueeze(1)
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        # Compute the TD errors and update the priorities
        td_errors = (expected_q_values - q_values).abs().squeeze().cpu().detach().numpy()
        for i, index in enumerate(indices):
            self.priorities[index] = (td_errors[i] + 1e-6) ** self.alpha
        # Compute the loss and update the network weights
        loss = (weights * self.loss_fn(q_values, expected_q_values)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Update the beta value and the number of steps
        self.beta = beta
        self.steps += 1

    def learn(self, num_episodes, max_steps, epsilon_start, epsilon_end, epsilon_decay, env):
        epsilon = epsilon_start
        rewards, epsilons = [], []
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            for step in range(max_steps):
                #env.render()
                action = self.act(state, epsilon)
                next_state, reward, done, _, info = env.step(action)
                total_reward += reward
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.optimize()
                if done:
                    break
            rewards.append(total_reward)
            epsilons.append(epsilon)
            epsilon = max(epsilon_end, epsilon_decay * epsilon)

            print(f"Episode {episode}/{num_episodes}: reward = {total_reward}, epsilon = {epsilon:.2f}, beta = {self.beta:.2f}")
        self.steps += 1  # increment steps
        self.beta = min(1, self.beta + (1 - self.beta) * self.steps / self.beta_annealing_steps)
        
        return rewards, epsilons
    
    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.q_net[-1].out_features - 1)
        else:
            state = torch.FloatTensor(state).to(self.device)
            q_values = self.q_net(state)
            action = q_values.max(-1)[1].item()
            q_values_list.append(q_values)
            return action

def train(agent, num_episodes, max_steps, epsilon_start, epsilon_end, epsilon_decay, env):
        epsilon = epsilon_start
        rewards, epsilons = [], []
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            for step in range(max_steps):
                #env.render()
                action = agent.act(state, epsilon)
                next_state, reward, done, _, info = env.step(action)
                total_reward += reward
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                agent.optimize()
                if done:
                    break
            rewards.append(total_reward)
            epsilons.append(epsilon)
            epsilon = max(epsilon_end, epsilon_decay * epsilon)
            print(f"Episode {episode}/{num_episodes}: reward = {total_reward}, epsilon = {epsilon:.2f}")
        return rewards, epsilons



env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
#agent = DQNAgent(obs_dim, n_actions, 16, 1e-3, 0.99, 100000, 8)
#rewards, epsilons = agent.learn(1000,500,1.0,0.01,.9955,env)

agent = PrioritizedDQNAgent(obs_dim, n_actions, 16, 1e-3, 0.99, 100000, 8, 0.6, 0.4, 1000000)
rewards, epsilons = agent.learn(500, 500, 1.0, 0.01, 0.9955, env)

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
#plot_q_values(q_values_list)