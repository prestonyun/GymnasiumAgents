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
            nn.Linear(64, 8),
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
            nn.Linear(8, 64),
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
        done = (reward.item() <= -1000.0)
        return next_state, reward, done

    def policy_value(self, state):
        return self.prediction(state)

class MuZeroAgent:
    def __init__(self, env, network, device, gamma=0.99):
        self.env = env
        self.network = network.to(device)
        self.device = device
        self.gamma = gamma

    def act(self, states, temperature=1.0):
        print('muzero act: ', states.shape)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        logits = self.network.policy_value(self.network.initial_states(states))
        logits = logits.view(-1, self.env.single_action_space.n)
        probabilities = torch.softmax(logits / temperature, dim=1)
        actions = torch.multinomial(probabilities, num_samples=1).squeeze().cpu().numpy()
        return actions
    
class Node:
    def __init__(self, agent, state, reward, done, parent=None, action=None):
        self.agent = agent
        self.state = state
        self.reward = reward.clone().detach().to(self.agent.device)
        self.done = done.clone().detach().to(self.agent.device)

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
        self.c = math.sqrt(2) # Exploration constant

    def run(self, state):
        if len(state.shape) == 1:
            state_tensor = state.clone().detach().unsqueeze(0).to(self.agent.device)
        else:
            state_tensor = state.clone().detach().to(self.agent.device)
        initial_states = self.agent.network.initial_states(state_tensor)
        root = Node(self.agent, initial_states, torch.tensor(0, dtype=torch.float32).to(self.agent.device), torch.tensor(False, dtype=torch.bool).to(self.agent.device))


        for _ in range(self.num_simulations):
            node, path = self.select_node(root)
            if not node.is_fully_expanded():
                reward = self.rollout(node)
            else:
                node.expand()
                reward = node.children[0].reward
            self.backpropagate(path, reward)

        return self.choose_best_action(root)

    def select_node(self, node):
        path = []
        while node.children:
            action, child = max(node.children.items(), key=lambda item: self.uct(item[1], node.visit_count))
            path.append((action, node))
            node = child
        return node, path


    def rollout(self, node):
        if node.is_terminal():
            return node.reward

        policy_logits = node.agent.network.policy_value(node.state)
        policy = F.softmax(policy_logits, dim=-1).view(-1).cpu().data.numpy()
        policy /= policy.sum()

        action = np.random.choice(len(policy), p=policy)

        if action not in node.children:
            action_tensor = torch.tensor([action], dtype=torch.float32).to(self.agent.device)
            next_state, reward, done = self.agent.network.next_states(node.state, action_tensor)
            child = Node(self.agent, next_state, reward, done, parent=node, action=action)
            node.children[action] = child

        return node.reward + self.agent.gamma * self.rollout(node.children[action])

    
    def choose_best_action(self, root):
        action, _ = max(root.children.items(), key=lambda item: item[1].visit_count)
        return action


    def backpropagate(self, path, reward):
        for action, node in path:
            node.visit_count += 1
            node.value_sum += reward
            node.value = node.value_sum / node.visit_count


    def uct(self, node, parent_visit_count):
        q = node.value_sum / node.visit_count if node.visit_count else 0
        u = self.c * math.sqrt(math.log(parent_visit_count) / (1 + node.visit_count))
        return q + u

class MCTSAgent(MuZeroAgent):
    def __init__(self, base_agent, device, num_simulations=50, gamma=0.99):
        super().__init__(base_agent.env, base_agent.network, device)
        self.num_simulations = num_simulations
        self.base_agent = base_agent

    def act(self, states, temperature=1.0, hidden_state=None):
        if type(states) is np.ndarray:
            states = torch.FloatTensor(states).to(self.device)
        else:
            states = torch.FloatTensor(states.clone().detach().cpu()).to(self.device)

        if hidden_state is None:
            hidden_state = self.network.initial_states(states)
        else:
            hidden_state = hidden_state.clone().detach().to(self.device)


        mcts = MCTS(self, self.num_simulations, self.base_agent.gamma)
        best_action = mcts.run(states)
        actions = np.array([best_action])

        return actions


def train_muzero(env_name, epochs, learning_rate, replay_buffer_size, num_envs, num_simulations, discount):
    env = gym.vector.make(env_name, num_envs=num_envs)
    input_shape = env.single_observation_space.shape
    action_space_size = env.single_action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = MuZeroNetwork(input_shape, action_space_size).to(device)
    agent = MuZeroAgent(env, network, device, gamma=0.99)
    mcts_agent = MCTSAgent(agent, device, num_simulations)

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)
    replay_buffer = deque(maxlen=replay_buffer_size)

    best_eval_reward = float('-inf')
    early_stopping_patience = 250
    early_stopping_counter = 0

    for epoch in range(epochs):
        # Generate data using MCTS and store it in the replay buffer
        states, _ = env.reset()
        dones = np.array([False] * num_envs)
        completed_episodes = 0

        while completed_episodes < num_envs:
            # actions = []
            for state in states:
                actions = mcts_agent.act(torch.tensor(states, dtype=torch.float32).to(device))
                next_states, rewards, new_dones, _, _ = env.step(actions)
            
            for state, action, reward, done, next_state, new_done in zip(states, actions, rewards, dones, next_states, new_dones):
                if not done:
                    replay_buffer.append((state, action, reward))
                    if new_done:
                        completed_episodes += 1

            states = next_states
            dones = new_dones

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
                loss += F.mse_loss(predicted_value.squeeze(0), target_value)

            loss.backward()
            optimizer.step()

            eval_reward = evaluate_agent(agent, env_name, num_episodes=5)
            print(f"Epoch {epoch}: Evalu reward = {eval_reward}, learning rate = {optimizer.param_groups[0]['lr']}, loss = {loss.item()}")

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

    return mcts_agent

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
        
        dones = False
        total_rewards = 0

        while not np.all(dones):
            env.render()
            actions = agent.act(states, temperature=0.001)
            states, rewards, dones, _, _ = env.step(actions)
            total_rewards += rewards

        print(f"Episode {episode + 1}: Total reward = {total_rewards}")
    env.close()

if __name__ == "__main__":
    env_name = "LunarLander-v2"
    epochs = 50
    learning_rate = 0.1
    replay_buffer_size = 2500

    agent = train_muzero(env_name, epochs, learning_rate, replay_buffer_size, 2, 50, 1)
    test_agent(agent, env_name, episodes=10)

    model_save_path = "muzero_model.pt"
    torch.save(agent.network.state_dict(), model_save_path)