import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


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

        # Define the weight matrix for the holographic memory
        self.weights = nn.Parameter(torch.randn(key_dim, value_dim, key_dim, value_dim, n_actions))

    def forward(self, key_vectors, value_vectors):
        batch_size = key_vectors.size(0)
        print(key_vectors.shape)
        print(value_vectors.shape)
        print(self.key_dim, self.value_dim)

        # Reshape the key and value vectors to prepare them for the holographic memory
        key_vectors = key_vectors.permute(1, 0, 2, 3).contiguous().view(1, -1, self.key_dim, self.value_dim)
        value_vectors = value_vectors.permute(1, 0, 2, 3).contiguous().view(1, -1, self.key_dim, self.value_dim)

        # Compute the key and value vectors using the self-attention mechanism
        key_vectors, _ = self.self_attn(key_vectors, value_vectors, value_vectors)

        # Compute the holographic memory vectors by taking the circular convolution of the key vectors with the weight matrices
        convolved = F.conv1d(key_vectors, self.weights, groups=self.n_actions)
        mem_vectors = torch.sum(convolved * value_vectors.unsqueeze(-1), dim=2)

        # Compute the Q-values by taking the inner product of the holographic memory vectors with the query vector
        q_values = torch.einsum("bln,blwa->blwa", key_vectors.squeeze(0), mem_vectors)

        return q_values

