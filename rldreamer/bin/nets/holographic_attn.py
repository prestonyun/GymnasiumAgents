import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from gymtest import TransformerRNN

class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=128):
        super(DuelingQNetwork, self).__init__()

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        # Define the Q-network architecture (e.g. using a hybrid transformer-RNN)
        self.transformer_rnn = TransformerRNN(obs_dim=obs_dim, key_dim=64, hidden_dim=hidden_dim, n_layers=2)
        self.fc_adv = nn.Linear(hidden_dim, n_actions)
        self.fc_val = nn.Linear(hidden_dim, 1)

        # Define the optimizer and loss function
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()

    def forward(self, obs, action=None):
        # Compute the Q-values for the given state-action pair
        key_vectors, value_vectors = self.transformer_rnn(obs)
        adv = F.relu(self.fc_adv(value_vectors))
        val = self.fc_val(value_vectors)
        q_values = val + adv - adv.mean(dim=1, keepdim=True)

        if action is not None:
            q_values = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        return q_values

class VectorHologram:
    def __init__(self, dim):
        self.dim = dim
        self.data = np.zeros((dim,), dtype=np.complex128)

    def add(self, vector):
        assert len(vector) == self.dim
        self.data += np.fft.fft(vector)

    def dot(self, vector):
        assert len(vector) == self.dim
        result = np.fft.ifft(self.data * np.fft.fft(vector))
        return np.real(result)

class HolographicAttentionNetwork(nn.Module):
    def __init__(self, obs_dim, key_dim, value_dim, n_heads, n_actions, dropout_prob=0.2):
        super(HolographicAttentionNetwork, self).__init__()
        self.n_heads = n_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.n_actions = n_actions

        # Define the self-attention mechanism
        self.self_attn = nn.MultiheadAttention(key_dim*value_dim, n_heads, dropout=dropout_prob)
        #self.self_attn = nn.MultiheadAttention(1, 1, dropout=dropout_prob)

        # Define the weight matrix for the holographic memory
        self.weights = nn.Parameter(torch.randn(key_dim, value_dim, key_dim, value_dim, n_actions))   

    def forward(self, key_vectors, value_vectors):
        batch_size = key_vectors.size(0)

        # Compute the holographic memory vectors by taking the circular convolution of the key vectors with the weight matrices
        convolved = F.conv1d(key_vectors.view(batch_size, self.key_dim * self.value_dim, -1), self.weights, groups=self.n_actions)
        mem_vectors = torch.sum(convolved.view(batch_size, self.n_actions, self.key_dim, self.value_dim, -1) * value_vectors.unsqueeze(1), dim=3)

        # Compute the Q-values by taking the inner product of the holographic memory vectors with the query vector
        q_values = torch.einsum("bln,blwa->blwa", key_vectors.squeeze(0), mem_vectors)

        return q_values


        """     
        def forward(self, key_vectors, value_vectors):
        batch_size = key_vectors.size(0)
        print(key_vectors.shape)
        print(value_vectors.shape)

        # Reshape the key and value vectors to prepare them for the holographic memory
        key_vectors = key_vectors.permute(1, 0, 2, 3).contiguous().view(1, batch_size, self.key_dim, self.value_dim)
        value_vectors = value_vectors.permute(1, 0, 2, 3).contiguous().view(1, -1, self.key_dim, self.value_dim)

        # Compute the key and value vectors using the self-attention mechanism
        key_vectors, _ = self.self_attn(key_vectors, value_vectors, value_vectors)

        # Compute the holographic memory vectors by taking the circular convolution of the key vectors with the weight matrices
        convolved = F.conv1d(key_vectors, self.weights, groups=self.n_actions)
        mem_vectors = torch.sum(convolved * value_vectors.unsqueeze(-1), dim=2)

        # Compute the Q-values by taking the inner product of the holographic memory vectors with the query vector
        q_values = torch.einsum("bln,blwa->blwa", key_vectors.squeeze(0), mem_vectors)

        return q_values """