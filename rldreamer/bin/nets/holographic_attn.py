import torch
import numpy as np

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
    def __init__(self, key_dim, value_dim, n_actions):
        super(HolographicAttentionNetwork, self).__init__()

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.n_actions = n_actions

        # Define the weight matrices for the holographic memory
        self.weights = nn.Parameter(torch.randn(key_dim, value_dim, n_actions))

    def forward(self, key_vectors, value_vectors):
        # Compute the holographic memory vectors by taking the circular convolution of the key vectors with the weight matrices
        mem_vectors = torch.sum(torch.einsum("bln,nka->bla", key_vectors, self.weights)[:, :, :, None] * value_vectors[:, None, :, :], dim=2)

        # Compute the Q-values by taking the inner product of the holographic memory vectors with the query vector
        q_values = torch.einsum("bln,bmn->blm", key_vectors, mem_vectors)

        return q_values
