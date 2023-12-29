import math
import torch
import torch.nn as nn
from torchkit import pytorch_utils as ptu
import torch.nn.functional as F


class ComplexGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super(ComplexGroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(ptu.ones(num_channels))
        self.bias = nn.Parameter(ptu.zeros(num_channels))

    def forward(self, X):
        """
        X: (batch_size, sequence_length, hidden_size)
        X is assumed to be complex
        """
        X = X.reshape(-1, self.num_groups, self.num_channels // self.num_groups)
        mean = X.mean(dim=2, keepdim=True)
        var = X.var(dim=2, keepdim=True)
        X = (X - mean) / torch.sqrt(var + self.eps)
        X = X.reshape(-1, self.num_channels)
        X = X * self.weight + self.bias

        return X


class ComplexLayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(ComplexLayerNorm, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(ptu.ones(num_channels))
        self.bias = nn.Parameter(ptu.zeros(num_channels))

    def forward(self, X):
        """
        X: unknown shape ending in hidden_size
        we treat the last dimension as the hidden_size
        """
        X_shape = X.shape
        X = X.reshape(-1, X_shape[-1])
        mean = X.mean(dim=1, keepdim=True)
        var = X.abs().var(dim=1, keepdim=True)
        X = (X - mean) / torch.sqrt(var + self.eps)
        X = X * self.weight + self.bias
        X = X.reshape(X_shape)
        return X


class ComplexFFN(nn.Module):
    """
    2 linear layers with no bias
    """

    def __init__(self, hidden_size, ffn_size):
        super(ComplexFFN, self).__init__()
        self.W1 = nn.Parameter(ptu.randn(hidden_size, ffn_size) / math.sqrt(hidden_size))
        self.W2 = nn.Parameter(ptu.randn(ffn_size, hidden_size) / math.sqrt(ffn_size))
        self.gelu = lambda x: 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

    def forward(self, X):
        """
        X: (batch_size, sequence_length, hidden_size)
        X is assumed to be complex
        """
        # reshaping
        X = X @ self.W1.to(X)
        X = self.gelu(X)
        X = X @ self.W2.to(X)

        return X


class SimpleRetention(nn.Module):
    def __init__(self, hidden_size, gamma, precision="single"):
        """
        Simple retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(SimpleRetention, self).__init__()

        if precision == "half":
            raise NotImplementedError("batchmm does not support half precision complex yet.")
            self.complex_type = torch.complex32
            self.real_type = torch.float16
        elif precision == "single":
            self.complex_type = torch.complex64
            self.real_type = torch.float32

        self.precision = precision
        self.hidden_size = hidden_size
        self.gamma = gamma

        self.i = torch.complex(torch.tensor(0.0), torch.tensor(1.0))

        self.W_Q = nn.Parameter(ptu.randn(hidden_size, hidden_size, dtype=self.real_type) / hidden_size)
        self.W_K = nn.Parameter(ptu.randn(hidden_size, hidden_size, dtype=self.real_type) / hidden_size)
        self.W_V = nn.Parameter(ptu.randn(hidden_size, hidden_size, dtype=self.real_type) / hidden_size)

        self.theta = ptu.randn(hidden_size) / hidden_size
        self.theta = nn.Parameter(self.theta)

    def forward(self, X):
        """
        Parallel (default) representation of the retention mechanism.
        X: (batch_size, sequence_length, hidden_size)
        """
        sequence_length = X.shape[1]
        D = self._get_D(sequence_length)

        if X.dtype != self.complex_type:
            X = torch.complex(X, ptu.zeros_like(X)).to(self.complex_type)

        i = self.i.to(X.device)
        ns = torch.arange(1, sequence_length + 1, dtype=self.real_type, device=X.device)
        ns = torch.complex(ns, ptu.zeros_like(ns)).to(self.complex_type)
        Theta = []
        for n in ns:
            Theta.append(torch.exp(i * n * self.theta))
        Theta = torch.stack(Theta, dim=0)
        Theta_bar = Theta.conj()
        Q = (X @ self.W_Q.to(self.complex_type)) * Theta.unsqueeze(0)
        K = (X @ self.W_K.to(self.complex_type)) * Theta_bar.unsqueeze(0)
        V = X @ self.W_V.to(self.complex_type)
        att = (Q @ K.permute(0, 2, 1)) * D.unsqueeze(0)

        return att @ V

    def forward_recurrent(self, x_n, s_n_1, n):
        """
        Recurrent representation of the retention mechanism.
        x_n: (batch_size, hidden_size)
        s_n_1: (batch_size, hidden_size)
        """
        if x_n.dtype != self.complex_type:
            x_n = torch.complex(x_n, ptu.zeros_like(x_n)).to(self.complex_type)

        n = torch.tensor(n, dtype=self.complex_type, device=x_n.device)

        Theta = torch.exp(self.i * n * self.theta)
        Theta_bar = Theta.conj()

        Q = (x_n @ self.W_Q.to(self.complex_type)) * Theta
        K = (x_n @ self.W_K.to(self.complex_type)) * Theta_bar
        V = x_n @ self.W_V.to(self.complex_type)

        # K: (batch_size, hidden_size)
        # V: (batch_size, hidden_size)
        # s_n_1: (batch_size, hidden_size, hidden_size)
        # s_n = gamma * s_n_1 + K^T @ V

        s_n = self.gamma * s_n_1 + K.unsqueeze(2) @ V.unsqueeze(1)

        return (Q.unsqueeze(1) @ s_n).squeeze(1), s_n

    def _get_D(self, sequence_length):
        # 下三角矩阵, 衰减
        D = ptu.zeros((sequence_length, sequence_length), dtype=self.real_type, requires_grad=False)
        for n in range(sequence_length):
            for m in range(sequence_length):
                if n >= m:
                    D[n, m] = self.gamma ** (n - m)
        return D.to(self.complex_type)


class MultiScaleRetention(nn.Module):
    def __init__(self, hidden_size, heads, precision="single"):
        """
        Multi-scale retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(MultiScaleRetention, self).__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.precision = precision
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = hidden_size // heads

        if precision == "half":
            raise NotImplementedError("batchmm does not support half precision complex yet.")
            self.complex_type = torch.complex32
            self.real_type = torch.float16
        elif precision == "single":
            self.complex_type = torch.complex64
            self.real_type = torch.float32

        self.gammas = (1 - torch.exp(
            torch.linspace(math.log(1 / 32), math.log(1 / 512), heads, dtype=self.real_type))).detach().cpu().tolist()

        self.swish = lambda x: x * torch.sigmoid(x)
        self.W_G = nn.Parameter(ptu.randn(hidden_size, hidden_size, dtype=self.complex_type) / hidden_size)
        self.W_O = nn.Parameter(ptu.randn(hidden_size, hidden_size, dtype=self.complex_type) / hidden_size)
        self.group_norm = ComplexGroupNorm(heads, hidden_size)

        self.retentions = nn.ModuleList([
            SimpleRetention(self.head_size, gamma) for gamma in self.gammas
        ])

    def forward(self, X):
        """
        parallel representation of the multi-scale retention mechanism
        """
        if X.dtype != self.complex_type:
            X = torch.complex(X, ptu.zeros_like(X)).to(self.complex_type)


        Y = []
        for i in range(self.heads):
            Y.append(self.retentions[i](X[:, :, i * self.head_size:(i + 1) * self.head_size]))

        Y = torch.cat(Y, dim=2)
        Y = self.group_norm(Y.reshape(-1, self.hidden_size)).reshape(X.shape)

        return (self.swish(X @ self.W_G) + Y) @ self.W_O


class RetNetEncoder(nn.Module):
    def __init__(self, context_dim, n_blocks, n_heads, is_agent=True):
        super(RetNetEncoder, self).__init__()
        self.layers = n_blocks
        self.hidden_dim = context_dim
        self.ffn_size = context_dim
        self.heads = n_heads

        self.retentions = nn.ModuleList([
            MultiScaleRetention(context_dim, n_heads)
            for _ in range(n_blocks)
        ])
        self.ffns = nn.ModuleList([
            ComplexFFN(context_dim, context_dim)
            for _ in range(n_blocks)
        ])
        self.layer_norm = ComplexLayerNorm(context_dim)
        self.is_agent = is_agent

    def forward(self, x):
        if not self.is_agent:
            meta_batch, batch_size, n_agents, context_dim = x.size()
            x = x.view(meta_batch * batch_size, n_agents, context_dim)
        else:
            meta_batch, batch_size, context_dim = x.size()
        for i in range(self.layers):
            y = self.retentions[i](self.layer_norm(x)) + x
            x = self.ffns[i](self.layer_norm(y)) + y
        o = x.mean(dim=1)
        if not self.is_agent:
            o = o.view(meta_batch, batch_size, context_dim)
        return o.real.float()








