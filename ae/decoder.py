# decoder只在训练时使用, 是集中式的, obs=state, action=action_n
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl
from torchkit import pytorch_utils as ptu
from torchkit.networks import Mlp, FlattenMlp


class TransitionDecoder(nn.Module):
    def __init__(self, context_dim, hidden_sizes, state_dim, action_n_dim,  dist="gaussian"):
        super().__init__()
        self.dist = dist
        output_dim = state_dim * 2 if self.dist == "gaussian" else state_dim
        self.mlp = Mlp(input_size=context_dim + state_dim + action_n_dim,
                       output_size=output_dim,
                       hidden_sizes=hidden_sizes,
                       hidden_activation=F.gelu)

    def forward(self, context, state, action):
        h = torch.cat((context, state, action), dim=-1)
        out = self.mlp(h)
        if self.dist == "gaussian":
            mu, log_std = torch.split(out, self.mlp.output_size // 2, dim=-1)
            return mu, log_std
        else:
            return out


class RewardDecoder(nn.Module):
    def __init__(self, context_dim, hidden_sizes, state_dim, action_n_dim, is_next_state=True, dist="gaussian"):
        super().__init__()
        self.is_next_state = is_next_state  # 是否使用next_state帮助预测reward
        self.dist = dist
        output_dim = 2 if self.dist == "gaussian" else 1
        if self.is_next_state:
            self.mlp = Mlp(input_size=context_dim + state_dim * 2 + action_n_dim,
                           output_size=output_dim,
                           hidden_sizes=hidden_sizes,
                           hidden_activation=F.gelu)
        else:
            self.mlp = Mlp(input_size=context_dim + state_dim + action_n_dim,
                           output_size=output_dim,
                           hidden_sizes=hidden_sizes,
                           hidden_activation=F.gelu)

    def forward(self, context, state, action, next_state=None):
        if self.is_next_state:
            h = torch.cat((context, state, action, next_state), dim=-1)
            out = self.mlp(h)
        else:
            h = torch.cat((context, state, action), dim=-1)
            out = self.mlp(h)
        if self.dist == "gaussian":
            mu, log_std = torch.split(out, self.mlp.output_size // 2, dim=-1)
            return mu, log_std
        else:
            return out


class TaskDecoder(nn.Module):
    def __init__(self, context_dim, hidden_sizes, n_tasks):
        super().__init__()
        self.mlp = Mlp(input_size=context_dim,
                       output_size=n_tasks,
                       hidden_sizes=hidden_sizes,
                       hidden_activation=F.gelu,
                       # output_activation=nn.Softmax(dim=-1)
                       )

    def forward(self, context):
        return self.mlp(context)
