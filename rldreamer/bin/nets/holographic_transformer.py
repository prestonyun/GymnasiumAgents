from typing import Tuple
from collections import deque
from .replaybuffer import ReplayBuffer

import copy
import random
import torch
import numpy as np
import torch.nn as nn

class HolographicEmbeddingLayer(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(HolographicEmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weights = torch.nn.Parameter(torch.randn(vocab_size, embedding_dim))

    def forward(self, input_indices):
        # Get the embeddings for the input indices
        input_embeddings = self.weights[input_indices]

        # Normalize the embeddings
        norm = torch.norm(input_embeddings, dim=1, keepdim=True)
        input_embeddings = input_embeddings / norm

        # Sum the normalized embeddings for all words in the input
        holo_embedding = torch.sum(input_embeddings, dim=0, keepdim=True)

        return holo_embedding

class GameStateDataset(torch.utils.data.Dataset):
    def __init__(self, game_states):
        self.game_states = game_states
        self.health_range = (0, 121)
        self.energy_range = (0, 100)
        self.location_range = (0, 10000)  # assuming location can be negative

    def __len__(self):
        return len(self.game_states)

    def __getitem__(self, idx):
        game_state = self.game_states[idx]

        # normalize and convert health to tensor
        health = game_state['health']
        health = np.clip(health, *self.health_range)  # clip to valid range
        health = (health - self.health_range[0]) / (self.health_range[1] - self.health_range[0])  # normalize
        health = torch.tensor(health, dtype=torch.float32)

        # normalize and convert energy to tensor
        energy = game_state['energy']
        energy = np.clip(energy, *self.energy_range)  # clip to valid range
        energy = (energy - self.energy_range[0]) / (self.energy_range[1] - self.energy_range[0])  # normalize
        energy = torch.tensor(energy, dtype=torch.float32)

        # normalize and convert location to tensor
        location = game_state['location']
        location = [np.clip(x, *self.location_range) for x in location]  # clip to valid range
        location = [(x - self.location_range[0]) / (self.location_range[1] - self.location_range[0]) for x in location]  # normalize
        location = torch.tensor(location, dtype=torch.float32)

        return health, energy, location

class Collator:
    def __init__(self, embedding):
        """Initialize the class.

        Args:
            embedding: The embedding layer.
        """
        self.embedding = embedding

    def __call__(self, batch):
        """Convert a batch of data into tensors.

        Args:
            batch: A list of data points.
        """
        # Unpack the batch into health, energy, and location.
        health, energy, location = zip(*batch)

        # Convert the data into tensors.
        health_tensor = torch.tensor(health)
        energy_tensor = torch.tensor(energy)
        location_tensor = torch.tensor(location)

        # Embed the data.
        health_embedded = self.embedding(health_tensor)
        energy_embedded = self.embedding(energy_tensor)
        location_embedded = self.embedding(location_tensor)
        
        # Return the embedded data.
        return health_embedded, energy_embedded, location_embedded


class DQN(torch.nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int):
        super(DQN, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class HolographicTransformer(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, hidden_dim, dropout_prob):
        super(HolographicTransformer, self).__init__()

        # Define the Holographic Embedding layer
        self.embedding = HolographicEmbeddingLayer(vocab_size, embedding_dim)

        # Define the Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_prob)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Define the output layer
        self.output_layer = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # Apply the Holographic Embedding layer to the input
        embedded_input = self.embedding(x)

        # Transpose the embedding tensor to prepare it for the Transformer Encoder
        embedded_input = embedded_input.transpose(0, 1)

        # Apply the Transformer Encoder to the embedded input
        encoded = self.encoder(embedded_input)

        # Transpose the encoded tensor back to its original shape
        encoded = encoded.transpose(0, 1)

        # Apply the output layer and return the result
        output = self.output_layer(encoded)
        return output.squeeze(-1)



def run_epoch(model, dataloader, optimizer, device):
    model.train()

    for batch in dataloader:
        # move batch to device
        batch = batch.to(device)

        # forward pass
        output = model(batch)

        # compute loss
        loss = compute_loss(output)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def process_game_state(game_state, embedding):
    health = game_state['health']
    energy = game_state['energy']
    location = game_state['location']

    game_state_tensor = torch.tensor([health, energy, location])
    game_state_embedding = embedding(game_state_tensor)

    return game_state_embedding

def compute_loss(model, optimizer, target_model, states, actions, rewards, next_states, dones, batch_size, gamma, device):
    # Convert states and next states to embeddings
    embedding = model.embedding
    state_embeddings = torch.stack([process_game_state(state, embedding) for state in states]).to(device)
    next_state_embeddings = torch.stack([process_game_state(state, embedding) for state in next_states]).to(device)

    # Compute Q-values for current states and next states
    current_Q = model(state_embeddings).gather(1, actions)
    next_Q = target_model(next_state_embeddings).max(1)[0].unsqueeze(1)
    target_Q = rewards + (gamma * next_Q * (1 - dones))

    # Compute loss
    loss = torch.nn.MSELoss()(current_Q, target_Q)

    # Backpropagate and update model parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train(agent, env, num_episodes, batch_size, gamma, eps_start, eps_end, eps_decay, target_update_frequency, device):
    replay_buffer = replaybuffer.ReplayBuffer(buffer_size=10000)

    target_agent = copy.deepcopy(agent)
    target_agent.to(device)

    optimizer = torch.optim.Adam(agent.parameters())

    eps = eps_start
    for episode in range(num_episodes):
        state = env.get_initial_state()
        done = False

        while not done:
            # choose action
            action = agent.act(state, eps)
            # take step in environment
            next_state, reward, done, info = env.step(action)
            # add experience to replay buffer
            replay_buffer.add(state, action, reward, next_state, done)

            # update agent
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                loss = compute_loss(agent, optimizer, target_agent, states, actions, rewards, next_states, dones, batch_size, gamma, device)
                update_target(agent, target_agent, target_update_frequency)

            state = next_state

            if done:
                replay_buffer.add(state, None, reward, None, done)

        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            loss = compute_loss(agent, optimizer, target_agent, states, actions, rewards, next_states, dones, gamma)
            update_target(agent, target_agent, target_update_frequency)

        # update epsilon for next episode
        eps = max(eps_end, eps_decay * eps)

def update_target(model, target_model):
    target_model.load_state_dict(model.state_dict())
