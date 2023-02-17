from collections import deque
import copy
import random
import torch

class HolographicEmbeddingLayer(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(HolographicEmbeddingLayer, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weights = torch.nn.Parameter(torch.randn(vocab_size, embedding_dim))

    def forward(self, input_indices):
        input_embeddings = self.weights[input_indices]
        norm = torch.sqrt(torch.sum(input_embeddings ** 2, dim=1, keepdim=True))
        input_embeddings = input_embeddings / norm
        holo_embedding = torch.sum(input_embeddings, dim=0, keepdim=True)

        return holo_embedding

class GameStateDataset(torch.utils.data.Dataset):
    def __init__(self, game_states):
        self.game_states = game_states

    def __len__(self):
        return len(self.game_states)

    def __getitem__(self, idx):
        game_state = self.game_states[idx]

        return game_state['health'], game_state['energy'], game_state['location']

class Collator:
    def __init__(self, embedding):
        self.embedding = embedding

    def __call__(self, batch):
        health, energy, location = zip(*batch)

        health_tensor = torch.tensor(health)
        energy_tensor = torch.tensor(energy)
        location_tensor = torch.tensor(location)

        health_embedded = self.embedding(health_tensor)
        energy_embedded = self.embedding(energy_tensor)
        location_embedded = self.embedding(location_tensor)
        
        return health_embedded, energy_embedded, location_embedded

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


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
    replay_buffer = ReplayBuffer(buffer_size=10000)

    target_agent = copy.deepcopy(agent)
    target_agent.to(device)

    optimizer = torch.optim.Adam(agent.parameters())

    eps = eps_start
    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            # choose action
            action = agent.act(state, eps)
            # take step in environment
            next_state, reward, done, info = env.step(action)
            # add experience to replay buffer
            replay_buffer.add(state, action, reward, next_state, done)

            state = next_state

            # update agent
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                loss = compute_loss(agent, optimizer, target_agent, states, actions, rewards, next_states, dones, batch_size, gamma, device)
                update_target(agent, target_agent, target_update_frequency)

        # update epsilon for next episode
        eps = max(eps_end, eps_decay * eps)




def update_target(model, target_model):
    target_model.load_state_dict(model.state_dict())
