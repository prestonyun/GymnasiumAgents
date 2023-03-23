import torch
import math
import torch.nn as nn
import torch.nn.functional as F
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

    def initial_state(self, x):
        return self.representation(x)

    def next_state(self, state, action):
        next_state = self.dynamics_state(torch.cat([state, action]))
        reward = self.dynamics_reward(torch.cat([state, action]))
        return next_state, reward

    def policy_value(self, state):
        return self.prediction(state)

class MuZeroAgent:
    def __init__(self, env, network, device):
        self.env = env
        self.network = network.to(device)
        self.device = device

    def act(self, state, temperature=1.0):
        state = torch.tensor(state.clone().detach(), dtype=torch.float32).to(self.device)
        logits = self.network.policy_value(self.network.initial_state(state))
        probabilities = torch.softmax(logits / temperature, dim=-1)
        action = torch.multinomial(probabilities, num_samples=1).squeeze().cpu().numpy()
        return action
    
class Node:
    def __init__(self, agent, state, reward, done, parent=None, action=None):
        self.agent = agent
        self.state = state
        self.reward = reward
        self.done = done

        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.value = 0

    def expand(self):
        if self.is_terminal():
            return

        action_space_size = self.agent.env.single_action_space.n
        policy_logits = self.agent.network.policy_value(self.state)
        policy = F.softmax(policy_logits, dim=1).squeeze(0).cpu().data.numpy()

        for action in range(action_space_size):
            action_tensor = torch.tensor([action], dtype=torch.float32).to(self.agent.device)
            next_state, reward, done = self.agent.network.next_states(self.state, action_tensor)
            self.children[action] = Node(self.agent, next_state, reward, done, parent=self, action=action)

    def is_terminal(self):
        return self.done

    def is_fully_expanded(self):
        return len(self.children) == self.agent.env.single_action_space.n

    def best_child(self, c_puct=1.0):
        ucb_values = [(action, self.ucb_value(child, c_puct)) for action, child in self.children.items()]
        best_action, _ = max(ucb_values, key=lambda x: x[1])
        return self.children[best_action]

    def ucb_value(self, child, c_puct):
        q_value = child.value / child.visit_count if child.visit_count != 0 else 0
        u_value = np.sqrt(self.visit_count) / (1 + child.visit_count)
        p_value = child.reward + child.agent.gamma * q_value
        return p_value + c_puct * u_value
    
class MCTS:
    def __init__(self, agent, num_simulations, discount):
        self.agent = agent
        self.num_simulations = num_simulations
        self.discount = discount

    def run(self, state):
        root = Node(self.agent.network.initial_state(state))

        for _ in range(self.num_simulations):
            node = self.select(root)
            if not node.children:
                reward = self.rollout(node)
            else:
                node.expand()
                reward = node.children[0].reward
            self.backpropagate(node, reward)

        return self.choose_best_action(root)

    def select(self, node):
        path = []
        while node.children:
            action, child = max(node.children.items(), key=lambda item: self.uct(item[1], node.visit_count))
            path.append((action, node))
            node = child
        return node, path

    def expand(self, node):
        state = node.state
        if not node.done:
            for action in range(self.agent.env.action_space.n):
                next_state, reward = self.agent.network.next_state(state, torch.tensor([action], device=self.agent.device))
                done = next_state.squeeze().eq(state.squeeze()).all().item()
                node.children[action] = Node(next_state, reward.item(), done)

    def rollout(self, node):
        while not node.is_terminal():
            legal_actions = node.legal_actions()

            if len(legal_actions) == 0:
                return 0

            action = self.agent.act(node.state.unsqueeze(0), temperature=0.5, hidden_state=node.state)[0]
            next_state, reward = self.agent.network.next_states(node.state, torch.tensor([action]))
            node = Node(next_state.squeeze(0), node, reward)

        return node.reward

    def backpropagate(self, path, reward):
        for action, node in path:
            node.visit_count += 1
            node.total_value += reward

    def uct(self, node, parent_visit_count):
        q = node.total_value / node.visit_count if node.visit_count else 0
        u = self.c * math.sqrt(math.log(parent_visit_count) / (1 + node.visit_count))
        return q + u

class MCTSAgent(MuZeroAgent):
    def __init__(self, base_agent, device, num_simulations=50):
        super().__init__(base_agent.env, base_agent.network, device)
        self.num_simulations = num_simulations
        self.base_agent = base_agent

    def act(self, state, temperature=1.0, hidden_state=None):
        state = torch.FloatTensor(state.clone().detach().cpu()).to(self.device)

        hidden_state = self.network.initial_state(state)

        logits = self.network.policy_value(hidden_state)
        probabilities = torch.softmax(logits / temperature, dim=-1)
        action = torch.multinomial(probabilities, num_samples=1).squeeze().cpu().numpy()

        return action

def train_muzero(env_name, epochs, learning_rate, replay_buffer_size, num_simulations, discount):
    env = gym.make(env_name)
    input_shape = env.observation_space.shape
    action_space_size = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = MuZeroNetwork(input_shape, action_space_size).to(device)
    agent = MuZeroAgent(env, network, device)
    agent = MCTSAgent(agent, device, num_simulations)

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.0125)
    replay_buffer = deque(maxlen=replay_buffer_size)

    best_eval_reward = float('-inf')
    early_stopping_patience = 250
    early_stopping_counter = 0

    for epoch in range(epochs):
        # Generate data using MCTS and store it in the replay buffer
        state, _ = env.reset()
        done = False

        while not done:
            action = agent.act(torch.tensor(state, dtype=torch.float32).to(device))
            next_state, reward, new_done, _, _ = env.step(action)
            
            replay_buffer.append((state, action, reward))

            state = next_state
            done = new_done

        # Train the network using data from the replay buffer
        if len(replay_buffer) == replay_buffer_size:
            optimizer.zero_grad()
            loss = 0

            for state, action, reward in replay_buffer:
                state = torch.tensor(state, dtype=torch.float32).to(device)
                action = torch.tensor(action, dtype=torch.long).to(device).unsqueeze(0)
                reward = torch.tensor(reward, dtype=torch.float32).to(device)

                # Compute target value
                with torch.no_grad():
                    next_state, _ = agent.network.next_state(agent.network.initial_state(state), action)
                    target_value = reward + agent.network.policy_value(next_state)

                # Compute predicted value
                predicted_value = agent.network.policy_value(agent.network.initial_state(state))

                # Compute loss and accumulate gradients
                loss += F.mse_loss(predicted_value, target_value)

            loss.backward()
            optimizer.step()

            eval_reward = evaluate_agent(agent, env_name)
            print(f"Epoch {epoch}: Evalu reward = {eval_reward}, learning rate = {optimizer.param_groups[0]['lr']}, loss = {loss.item()}")

            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                print('new best!')
                torch.save(network.state_dict(), "best_muzero_model.pt")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping")
                    break
            
            lr_scheduler.step(eval_reward)
            # Logging
            print(f"Epoch {epoch}: Loss = {loss.mean()}")

    return agent

def evaluate_agent(agent, env_name):
    env = gym.make(env_name)
    total_reward = 0
    state, _ = env.reset()
    done = False

    while not done:
        action = agent.act(torch.tensor(state, dtype=torch.float32), temperature=0.001)
        state, reward, new_done, _, _ = env.step(action)
        total_reward += reward
        done = new_done

    return total_reward

def test_agent(agent, env_name, episodes):
    env = gym.make(env_name, render_mode="human")

    for episode in range(episodes):

        state, _ = env.reset()
        env.render()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(torch.tensor(state, dtype=torch.float32), temperature=0.001)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward

        print(f"Episode {episode + 1}: Total reward = {total_reward}")
    env.close()

def load_model(model_path):
    env_name = "LunarLander-v2"
    env = gym.make(env_name)
    input_shape = env.observation_space.shape
    action_space_size = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = MuZeroNetwork(input_shape, action_space_size).to(device)
    network.load_state_dict(torch.load(model_path))
    agent = MuZeroAgent(env, network, device)
    agent = MCTSAgent(agent, device, num_simulations=50)

    return agent

if __name__ == "__main__":
    env_name = "LunarLander-v2"
    epochs = 2000
    learning_rate = 0.1
    replay_buffer_size = 1000
    agent = load_model("muzero_model.pt")
    agent = train_muzero(env_name, epochs, learning_rate, replay_buffer_size, 50, 1)
    test_agent(agent, env_name, episodes=10)

    model_save_path = "muzero_model.pt"
    torch.save(agent.network.state_dict(), model_save_path)