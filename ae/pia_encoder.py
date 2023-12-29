import torch
import torch.nn as nn
from torch.nn import functional as F

from torchkit import pytorch_utils as ptu


class PIAEncoder(nn.Module):
    def __init__(self, context_dim, is_agent=True):
        super(PIAEncoder, self).__init__()
        self.context_dim = context_dim
        self.W_a = ptu.m_init(nn.Linear(context_dim, 1))
        self.W_v = ptu.m_init(nn.Linear(context_dim, context_dim))
        self.W_o = ptu.m_init(nn.Linear(context_dim, context_dim))
        self.is_agent = is_agent

    def forward(self, x):
        if not self.is_agent:
            meta_batch, batch_size, n_agents, context_dim = x.size()
            x = x.view(meta_batch * batch_size, n_agents, context_dim)
        else:
            meta_batch, batch_size, context_dim = x.size()

        attn = self.W_a(x).transpose(-1, -2)
        attn = F.softmax(attn, dim=-1)

        x = self.W_v(x)
        output = torch.matmul(attn, x).squeeze(1)
        output = self.W_o(output)
        if not self.is_agent:
            output = output.view(meta_batch, batch_size, self.context_dim)
        return output