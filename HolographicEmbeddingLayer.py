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

def compute_loss(model, optimizer, target_model, replay_buffer, batch_size, gamma, device):
    # Sample a batch from the replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Convert states and next states to embeddings
    state_embeddings = torch.stack([process_game_state(state, model) for state in states]).to(device)
    next_state_embeddings = torch.stack([process_game_state(state, model) for state in next_states]).to(device)

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
