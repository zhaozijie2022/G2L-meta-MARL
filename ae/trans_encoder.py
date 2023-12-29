import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchkit import pytorch_utils as ptu
from torchkit.networks import Mlp, FlattenMlp


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(SelfAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model should be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_q = ptu.m_init(nn.Linear(d_model, d_model))
        self.W_k = ptu.m_init(nn.Linear(d_model, d_model))
        self.W_v = ptu.m_init(nn.Linear(d_model, d_model))
        self.W_o = ptu.m_init(nn.Linear(d_model, d_model))


    def forward(self, x):
        batch_size, seq_len, _ = x.size()  # batch_size: Batch size, seq_len: Sequence length

        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)

        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)
        return output


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, max_len=1000):
        super().__init__()
        self.P = torch.zeros((1, max_len, num_hiddens))
        # 在位置嵌⼊矩阵P中，⾏代表词元在序列中的位置，列代表位置编码的不同维度
        position = torch.arange(0, max_len, dtype=torch.float32).reshape(-1, 1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, num_hiddens, 2).float() * (-np.log(10000.0) / num_hiddens))  # (num_hiddens / 2, )
        self.P[:, :, 0::2] = torch.sin(position * div_term)  # 偶数列
        self.P[:, :, 1::2] = torch.cos(position * div_term)  # 奇数列

    def forward(self, x):
        # x.shape: (batch_size, seq_len, num_hiddens)
        return x + self.P[:, :x.shape[1], :].to(x.device)


class EncodeBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super(EncodeBlock, self).__init__()
        self.attn = SelfAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = Mlp(d_model, d_model, [d_model], hidden_activation=F.gelu)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ffn(x))
        return x


class TransEncoder(nn.Module):
    def __init__(self, context_dim, n_blocks, n_heads, is_agent=True):
        super(TransEncoder, self).__init__()
        self.context_dim = context_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads

        self.pe = PositionalEncoding(context_dim)
        self.ln = nn.LayerNorm(context_dim)
        self.blocks = nn.Sequential(*[EncodeBlock(context_dim, n_heads) for _ in range(n_blocks)])
        self.is_agent = is_agent

    def forward(self, x):
        if not self.is_agent:
            meta_batch, batch_size, n_agents, context_dim = x.size()
            x = x.view(meta_batch * batch_size, n_agents, context_dim)
            # output = self.blocks(self.ln(self.pe(x))).mean(dim=1)
            output = self.blocks(self.ln(self.pe(x)))[:, 0]
            return output.view(meta_batch, batch_size, self.context_dim)
        else:
            # return self.blocks(self.ln(x)).mean(dim=1)
            return self.blocks(self.ln(x))[:, 0]


















