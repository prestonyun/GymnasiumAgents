import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

import gymnasium as gym


class MuZeroNetwork(nn.Module):
    def __init__(self, input_shape, action_space_size):
        super(MuZeroNetwork, self).__init__()
        self.representation = nn.Sequential(
            nn.Linear(np.prod(input_shape), 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.dynamics_state = nn.Sequential(
            nn.Linear(64 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.dynamics_reward = nn.Sequential(
            nn.Linear(64 + 1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        self.prediction = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space_size)
        )

    def initial_states(self, x):
        return self.representation(x)

    def next_states(self, state, action):
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        next_state = self.dynamics_state(torch.cat([state, action], dim=1))
        reward = self.dynamics_reward(torch.cat([state, action], dim=1))
        return next_state, reward

    def policy_value(self, state):
        return self.prediction(state)

class MuZeroAgent:
    def __init__(self, env, network, device):
        self.env = env
        self.network = network.to(device)
        self.device = device

    def act(self, states, temperature=1.0):
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        logits = self.network.policy_value(self.network.initial_states(states))
        logits = logits.view(-1, self.env.single_action_space.n)
        probabilities = torch.softmax(logits / temperature, dim=1)
        actions = torch.multinomial(probabilities, num_samples=1).squeeze().cpu().numpy()
        return actions

def train_muzero(env_name, epochs, learning_rate, replay_buffer_size, num_envs):
    env = gym.vector.make(env_name, num_envs)
    input_shape = env.single_observation_space.shape
    action_space_size = env.single_action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = MuZeroNetwork(input_shape, action_space_size).to(device)
    agent = MuZeroAgent(env, network, device)
    
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)
    replay_buffer = deque(maxlen=replay_buffer_size)

    best_eval_reward = float('-inf')
    early_stopping_patience = 20
    early_stopping_counter = 0

    for epoch in range(epochs):
        # Generate data using MCTS and store it in the replay buffer
        states, _ = env.reset()
        dones = np.array([False] * num_envs)
        completed_episodes = 0

        while completed_episodes < num_envs:
            actions = agent.act(states)
            # print(action)
            next_states, rewards, new_dones, _, _ = env.step(actions)
            
            for state, action, reward, done, next_state, new_done in zip(states, actions, rewards, dones, next_states, new_dones):
                if not done:
                    replay_buffer.append((state, action, reward))
                    if new_done:
                        completed_episodes += 1

            states = next_states
        
        print(f"Epoch {epoch}")

        # Train the network using data from the replay buffer
        if len(replay_buffer) == replay_buffer_size:
            optimizer.zero_grad()
            loss = 0

            for state, action, reward in replay_buffer:
                state = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
                action = torch.tensor([action], dtype=torch.long).to(device)
                reward = torch.tensor([reward], dtype=torch.float32).to(device)

                # Compute target value
                with torch.no_grad():
                    next_state, _ = agent.network.next_states(agent.network.initial_states(state), action)
                    target_value = reward + agent.network.policy_value(next_state).max().unsqueeze(0)


                # Compute predicted value
                predicted_value = agent.network.policy_value(agent.network.initial_states(state)).gather(1, action.unsqueeze(1))

                # Compute loss and accumulate gradients
                loss += nn.functional.mse_loss(predicted_value.squeeze(0), target_value)

            loss.backward()
            optimizer.step()

            eval_reward = evaluate_agent(agent, env_name, num_episodes=5)
            print(f"Epoch {epoch}: Evalu reward = {eval_reward}")

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save(network.state_dict(), "best_muzero_model.pt")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping")
                    break
            
            lr_scheduler.step(eval_reward)
            # Logging
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item()}")

    return agent

def evaluate_agent(agent, env_name, num_episodes):
    env = gym.vector.make(env_name, num_episodes)
    total_rewards = np.zeros(num_episodes)
    states, _ = env.reset()
    dones = np.array([False] * num_episodes)

    while not np.all(dones):
        actions = agent.act(states, temperature=0.001)
        states, rewards, new_dones, _, _ = env.step(actions)
        total_rewards += rewards * ~dones
        dones = np.logical_or(dones, new_dones)

    return total_rewards.mean()

def test_agent(agent, env_name, episodes):
    env = gym.make(env_name, render_mode="human")

    for episode in range(episodes):

        states, _ = env.reset()
        env.render()
        dones = False
        total_rewards = 0

        while not np.all(dones):
            actions = agent.act(states, temperature=0.001)
            states, rewards, dones, _, _ = env.step(actions)
            total_rewards += rewards

        print(f"Episode {episode + 1}: Total reward = {total_rewards}")
    env.close()

if __name__ == "__main__":
    env_name = "LunarLander-v2"
    epochs = 500
    learning_rate = 0.01
    replay_buffer_size = 1000

    agent = train_muzero(env_name, epochs, learning_rate, replay_buffer_size, 4)
    test_agent(agent, env_name, episodes=10)

    model_save_path = "muzero_model.pt"
    torch.save(agent.network.state_dict(), model_save_path)