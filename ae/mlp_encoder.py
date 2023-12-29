import torch
import torch.nn as nn
import torch.nn.functional as F
from torchkit import pytorch_utils as ptu
from torchkit.networks import Mlp, FlattenMlp


# 单步单智能体编码器, (o_i^t, a_i^t, r_i^t, o_i^{t+1}) -> z_i^t
class MLPEncoder(nn.Module):
    def __init__(self, obs_dim, action_dim, reward_dim=1, done_dim=0,
                 hidden_sizes=[64], context_dim=32, normalize=False,
                 output_activation=ptu.identity, **kwargs):
        super(MLPEncoder, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.done_dim = done_dim
        self.use_done = True if done_dim else False
        self.latent_dim = context_dim

        self.encoder = Mlp(input_size=obs_dim * 2 + action_dim + reward_dim + done_dim,
                           output_size=context_dim,
                           hidden_sizes=hidden_sizes,
                           hidden_activation=F.gelu,
                           output_activation=output_activation,)

        self.normalize = normalize

    def forward(self, obs, action, reward, next_obs, done=None):
        # (batch_size, xxx_dim) or (meta_batch, batch_size, xxx_dim)
        if self.use_done:
            f_input = torch.cat([obs, action, reward, next_obs, done], dim=-1)
            out = self.encoder(f_input)
        else:
            f_input = torch.cat([obs, action, reward, next_obs], dim=-1)
            out = self.encoder(f_input)
        return F.normalize(out) if self.normalize else out


class LocalEncoder(nn.Module):
    def __init__(self, context_dim=32, hidden_sizes=[64], normalize=False,
                 output_activation=ptu.identity, **kwargs):
        super(LocalEncoder, self).__init__()

        self.latent_dim = context_dim

        self.encoder = Mlp(input_size=context_dim,
                           output_size=context_dim,
                           hidden_sizes=hidden_sizes,
                           hidden_activation=F.gelu,
                           output_activation=output_activation,)

        self.normalize = normalize

    def forward(self, z):
        out = self.encoder(z)
        return F.normalize(out) if self.normalize else out