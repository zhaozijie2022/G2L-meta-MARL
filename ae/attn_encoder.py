import torch
import torch.nn as nn
from torch.nn import functional as F

from torchkit import pytorch_utils as ptu


class SelfAttnEncoder(nn.Module):
    def __init__(self, context_dim, n_heads, is_agent=True):
        super(SelfAttnEncoder, self).__init__()
        assert context_dim % n_heads == 0, "d_model should be divisible by n_heads"

        self.context_dim = context_dim
        self.n_heads = n_heads
        self.head_dim = context_dim // n_heads

        self.W_q = ptu.m_init(nn.Linear(context_dim, context_dim))
        self.W_k = ptu.m_init(nn.Linear(context_dim, context_dim))
        self.W_v = ptu.m_init(nn.Linear(context_dim, context_dim))
        self.W_o = ptu.m_init(nn.Linear(context_dim, context_dim))

        self.is_agent = is_agent

    def forward(self, x):
        if not self.is_agent:
            meta_batch, batch_size, n_agents, context_dim = x.size()
            x = x.view(meta_batch * batch_size, n_agents, context_dim)
        else:
            meta_batch, batch_size, context_dim = x.size()
        B, L, _ = x.size()  # B: Batch size, L: Sequence length

        # (meta_batch, ep_len, context_dim) -> (meta_batch, n_heads, ep_len, head_dim)
        q = self.W_q(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(B, L, self.context_dim)
        # output = self.W_o(output).mean(dim=1)
        output = self.W_o(output)[:, 0]
        if not self.is_agent:
            output = output.view(meta_batch, batch_size, self.context_dim)
        return output
