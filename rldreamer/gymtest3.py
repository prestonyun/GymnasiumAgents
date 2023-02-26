import random
import torch

import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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
        self.replay_buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def learn(self, num_episodes, max_steps, epsilon_start, epsilon_end, epsilon_decay, env):
        epsilon = epsilon_start
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0
            for step in range(max_steps):
                action = self.act(state, epsilon)
                next_state, reward, done, _, info = env.step(action)
                total_reward += reward
                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.optimize()
                if done:
                    break
            epsilon = max(epsilon_end, epsilon_decay * epsilon)
            print(f"Episode {episode}/{num_episodes}: reward = {total_reward}, epsilon = {epsilon:.2f}")


env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
agent = DQNAgent(obs_dim, n_actions, 128, 1e-3, 0.99, 10000, 32)
agent.learn(1000,500,1.0,0.01,0.995,env)