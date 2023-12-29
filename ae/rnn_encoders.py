import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl
from torchkit import pytorch_utils as ptu
from torchkit.networks import Mlp, FlattenMlp


# 多智能体单步, z_i^t -> z^t  (meta_batch, ep_len, context_dim) -> (meta_batch, context_dim)
# 多智能体多步, z^t -> z

class RNNEncoder(nn.Module):
    def __init__(self, context_dim, is_agent=True):
        super(RNNEncoder, self).__init__()
        self.context_dim = context_dim
        self.is_agent = is_agent

        self.fc1 = nn.Linear(context_dim, context_dim)
        self.rnn = nn.GRU(context_dim, context_dim, 1, batch_first=True)
        self.fc2 = nn.Linear(context_dim, context_dim)

    def forward(self, x):
        # 1.多智能体单步, 输入为(meta_batch, batch_size, n_agents, context_dim)
        # 2.多智能体多步, 输入为(meta_batch, batch_size, context_dim)
        if not self.is_agent:
            meta_batch, batch_size, n_agents, context_dim = x.size()
            x = x.view(meta_batch * batch_size, n_agents, context_dim)
        else:
            meta_batch, batch_size, context_dim = x.size()

        h = F.gelu(self.fc1(x))  # (B, n_agents or seq_len, context_dim)
        o, new_h = self.rnn(h)
        o = F.gelu(self.fc2(o))
        o = o[:, -1, :]
        if not self.is_agent:
            o = o.view(meta_batch, batch_size, self.context_dim)
        return o

    def seq(self, x):
        return self.forward(x)

    def one_step(self, x, h):
        h = F.gelu(self.fc1(x))
        o, new_h = self.rnn(h, h)
        o = F.gelu(self.fc2(o))
        return o, new_h


class MATERNNEncoder(nn.Module):
    # used for ind MATE
    def __init__(self, obs_dim, action_dim, reward_dim=1, done_dim=0,
                 hidden_dim=128, context_dim=32,
                 n_hidden_bfr=1, n_hidden_aft=1, normalize=False):
        super(MATERNNEncoder, self).__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        self.obs_encoder = utl.FeatureExtractor(obs_dim, 2 * obs_dim, F.relu)
        self.action_encoder = utl.FeatureExtractor(action_dim, 2 * action_dim, F.relu)
        self.reward_encoder = utl.FeatureExtractor(reward_dim, 2 * reward_dim, F.relu)

        # 2. fc before gru
        # curr_input_dim = (obs_dim + action_dim + reward_dim) * 2
        # self.fc_before_gru = nn.ModuleList([])
        # for i in range(n_hidden_bfr):
        #     self.fc_before_gru.append(nn.Linear(curr_input_dim, hidden_dim))

        # 3. gru
        self.gru = nn.GRU(input_size=(obs_dim + action_dim + reward_dim) * 2,
                          hidden_size=hidden_dim,
                          num_layers=1,
                          batch_first=True)  # batch_first=True, input/output shape: (batch, seq_len, input_dim)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        # # 4. fc after gru
        # self.fc_after_gru = nn.ModuleList([])
        # for i in range(n_hidden_aft):
        #     self.fc_after_gru.append(nn.Linear(hidden_dim, hidden_dim))


        # 5. output
        self.fc_last = nn.Linear(hidden_dim, context_dim)

    def seq(self, obs, action, reward):  # -> (rl_batch_size, seq_len, context_dim)
        # sequential forward, input_size: (batch, seq_len, xxx_dim)
        # 输入一个序列, 输出在每个时间步上的context, 主要用于训练
        h = torch.cat([self.obs_encoder(obs),
                       self.action_encoder(action),
                       self.reward_encoder(reward)], dim=-1)

        # for fc in self.fc_before_gru:
        #     h = F.relu(fc(h))
        out, new_h = self.gru(h)  # out.shape: (batch, seq_len, hidden_dim)
        # for fc in self.fc_after_gru:
        #     output = F.relu(fc(output))

        # task_mu = self.fc_mu(output)
        # task_logstd = self.fc_logstd(output)
        out = self.fc_last(out)

        return out, new_h

    def one_step(self, obs, action, reward, hidden_state):
        # 输入一个时间步的obs, action, reward, 输出下一个时间步的hidden_state, 主要用于测试
        h = torch.cat([self.obs_encoder(obs),
                       self.action_encoder(action),
                       self.reward_encoder(reward)], dim=-1)

        # for fc in self.fc_before_gru:
        #     h = F.relu(fc(h))

        h, new_h = self.gru(h, hidden_state)

        # for fc in self.fc_after_gru:
        #     h = F.relu(fc(h))
        #
        # task_mu = self.fc_mu(h)
        # task_logstd = self.fc_logstd(h)
        out = self.fc_last(h)

        return out, new_h

    # def sample(self, task_mu, task_logstd):
    #     # sample from task_mu, task_logstd
    #     task_std = torch.exp(task_logstd)
    #     task_sample = torch.randn_like(task_mu) * task_std + task_mu
    #     return task_sample

    # def prior(self, batch_size):  # -> (rl_batch_size, 1, xxx_dim)
    #     # 先验, 即没有输入obs, action, reward时的输出
    #     h = ptu.zeros((batch_size, 1, self.hidden_dim), requires_grad=True)
    #     for i in range(len(self.fc_after_gru)):
    #         h = F.relu(self.fc_after_gru[i](h))
    #     task_mean = self.fc_mu(h)
    #     task_logstd = self.fc_logstd(h)
    #     hidden = ptu.zeros((1, batch_size, self.hidden_dim), requires_grad=True)
    #     return task_mean, task_logstd, hidden











