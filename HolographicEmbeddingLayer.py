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

        #tensor = process_game_state(game_state)
        return game_state #tensor

class Collator:
    def __init__(self, model):
        self.model = model

    def __call__(self, batch):
        tensor = torch.stack(batch)
        tensor = self.model.holographic_embedding(tensor)
        
        return tensor