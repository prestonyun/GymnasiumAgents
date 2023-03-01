import random
import torch

import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import Tensor


class PrioritizedDQNAgent:
    """
    A prioritized Deep Q-Network (DQN) agent for solving reinforcement learning problems.

    Args:
        state_size (int): The number of dimensions in the state space.
        action_size (int): The number of possible actions in the action space.
        hidden_size (int): The number of neurons in each hidden layer of the Q-network.
        lr (float): The learning rate for the optimizer.
        gamma (float): The discount factor for future rewards.
        capacity (int): The maximum size of the replay buffer.
        batch_size (int): The size of the batch used for each optimization step.
        alpha (float): The exponent used for computing the priorities of transitions in the replay buffer.
        beta_start (float): The initial value of the exponent used for computing the importance sampling weights of transitions in the replay buffer.
        beta_annealing_steps (int): The number of steps over which to anneal the beta value from its initial value to 1.

    Attributes:
        device (torch.device): The device used for training the neural network.
        q_net (nn.Sequential): The neural network used for estimating Q-values.
        optimizer (torch.optim): The optimizer used for updating the Q-network.
        batch_size (int): The size of the batch used for each optimization step.
        gamma (float): The discount factor for future rewards.
        loss_fn (nn.MSELoss): The loss function used for computing the optimization loss.
        capacity (int): The maximum size of the replay buffer.
        replay_buffer (list): The replay buffer used for storing experiences.
        priority_pos (int): The position in the replay buffer to add the next experience.
        priorities (np.ndarray): The priority values of experiences in the replay buffer.
        alpha (float): The exponent used for computing the priorities of transitions in the replay buffer.
        beta (float): The exponent used for computing the importance sampling weights of transitions in the replay buffer.
        beta_annealing_steps (int): The number of steps over which to anneal the beta value from its initial value to 1.
        steps (int): The number of steps taken during training.

    Methods:
        remember(state, action, reward, next_state, done):
            Adds a new experience tuple to the replay buffer and updates the priorities based on the TD error of the transition.
        sample(beta=None):
            Samples a batch of transitions from the replay buffer based on their priorities, computes their importance sampling weights, and returns them as tensors.
        optimize():
            Performs a single optimization step on a batch of transitions sampled from the replay buffer using the Q-learning algorithm and the prioritized replay buffer.
        learn(num_episodes, max_steps, epsilon_start, epsilon_end, epsilon_decay, env):
            Trains the agent for a specified number of episodes using the prioritized DQN algorithm and returns the total reward and epsilon values for each episode.
        act(state, epsilon):
            Computes an action for a given state using the Q-network and an epsilon-greedy exploration strategy.
    """
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
        """
        Adds a new transition to the replay buffer and updates the priorities based on the TD error of the transition.

        Args:
            state (np.ndarray): The current state.
            action (int): The action taken in the current state.
            reward (float): The reward received for taking the action.
            next_state (np.ndarray): The resulting state after taking the action.
            done (bool): Whether the episode has ended after taking the action.

        Returns:
            None
        """
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
        """
        Samples a batch of transitions from the replay buffer based on their priorities and computes the importance sampling weights.

        Args:
            beta (float): The degree to which to adjust the probabilities of selecting each transition based on their priorities.

        Returns:
            A tuple containing the following numpy arrays:
            - states: The current states in the sampled transitions.
            - actions: The actions taken in the current states.
            - rewards: The rewards received for taking the actions.
            - next_states: The resulting states after taking the actions.
            - dones: Whether the episodes have ended after taking the actions.
            - weights: The importance sampling weights for the sampled transitions.
            - indices: The indices of the sampled transitions in the replay buffer.
        """
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
        """
        Performs a single optimization step on a batch of transitions sampled from the replay buffer.

        Returns:
            None
        """
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
        """
        Trains the agent for a specified number of episodes.

        Args:
            num_episodes (int): The number of episodes to train the agent for.
            max_steps (int): The maximum number of steps per episode.
            epsilon_start (float): The initial exploration rate.
            epsilon_end (float): The minimum exploration rate.
            epsilon_decay (float): The rate at which the exploration rate decays.
            env (gym.Env): The environment to train the agent in.

        Returns:
            tuple: A tuple containing two lists. The first list contains the total rewards obtained by the agent for each episode.
                The second list contains the exploration rates used by the agent for each episode.
        """
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
        """
        Selects an action to take based on the given state and exploration rate.

        Args:
        - state (array-like): The current state.
        - epsilon (float): The exploration rate. A value between 0 and 1.

        Returns:
        - int: The index of the action to take.
        """
        if random.random() < epsilon:
            return random.randint(0, self.q_net[-1].out_features - 1)
        else:
            state = torch.FloatTensor(state).to(self.device)
            q_values = self.q_net(state)
            action = q_values.max(-1)[1].item()
            return action

def plot_learning_curve(rewards, epsilons):
    """
    Plots the learning curve of a RL agent during training, showing the total reward obtained in each episode and the epsilon value used for exploration.

    Args:
        rewards (list): A list of total rewards obtained in each episode during training.
        epsilons (list): A list of epsilon values used for exploration in each episode during training.
        
    Returns:
        None. The plot is displayed using Matplotlib.
    """
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.twinx()
    plt.plot(epsilons, color='r')
    plt.ylabel('Epsilon')
    plt.show()

def main():
    """
    Trains a PrioritizedDQNAgent on the CartPole-v1 environment in Gym using the learn method, and plots the learning
    curve for the trained agent.

    Returns:
        None
    """
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = PrioritizedDQNAgent(obs_dim, n_actions, 16, 1e-3, 0.99, 100000, 8, 0.6, 0.4, 10000000)
    rewards, epsilons = agent.learn(500, 1000, 1.0, 0.01, 0.9955, env)

    plot_learning_curve(rewards, epsilons)


if __name__ == "__main__":
    main()