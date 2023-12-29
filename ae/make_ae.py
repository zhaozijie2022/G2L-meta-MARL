# repeat() 不会造成梯度的累加
import os
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW

from ae.pia_encoder import PIAEncoder
from ae.attn_encoder import SelfAttnEncoder
from ae.mlp_encoder import MLPEncoder, LocalEncoder
from ae.retnet_encoder import RetNetEncoder
from ae.rnn_encoders import RNNEncoder
from ae.trans_encoder import TransEncoder
from torchkit import pytorch_utils as ptu
from torchkit.networks import Mlp
from utils.util import update_linear_schedule
import pickle


def make_ae(cfg):
    return ContextEncoder(cfg=cfg).to(ptu.device)


class BaseEncoder(nn.Module):
    def __init__(self, *kwargs):
        super(BaseEncoder, self).__init__()

    def prior(self, meta_batch):
        return torch.randn(meta_batch, self.n_agents, self.context_dim).to(ptu.device)

    def kl_div(self, task_mu, task_log_std):
        task_std = F.softplus(task_log_std)
        return 0.5 * (task_mu ** 2 + task_std ** 2 - 2 * task_log_std - 1).mean()


class ContextEncoder(BaseEncoder):
    def __init__(self, cfg):
        super(ContextEncoder, self).__init__()
        self.cfg = cfg
        self.obs_dim_n = cfg.obs_dim_n
        self.action_dim_n = cfg.action_dim_n
        self.n_agents = cfg.n_agents
        self.context_dim = cfg.context_dim
        self.state_dim = cfg.state_dim
        self.n_tasks = cfg.n_tasks
        self.lr = cfg.lr
        self.encoder_type = cfg.ae_type
        self.kl_weight = cfg.kl_weight

        # region Leveled Task Encoder
        self.transition_encoders = [
            MLPEncoder(
                obs_dim=self.obs_dim_n[i],
                action_dim=self.action_dim_n[i],
                context_dim=self.context_dim,
                hidden_sizes=cfg.hidden_sizes_mlp,
            ).to(ptu.device)
            for i in range(self.n_agents)]
        if self.encoder_type == "self_attn":
            self.temporal_encoder = SelfAttnEncoder(context_dim=self.context_dim, n_heads=4,
                                                    is_agent=False).to(ptu.device)
            self.agent_encoder = SelfAttnEncoder(context_dim=self.context_dim, n_heads=4,
                                                 is_agent=True).to(ptu.device)
        elif self.encoder_type == "pia":
            self.temporal_encoder = PIAEncoder(context_dim=self.context_dim, is_agent=False).to(ptu.device)
            self.agent_encoder = PIAEncoder(context_dim=self.context_dim, is_agent=True).to(ptu.device)
        elif self.encoder_type == "transformer":
            self.temporal_encoder = TransEncoder(context_dim=self.context_dim, n_blocks=cfg.n_blocks, n_heads=4,
                                                 is_agent=False).to(ptu.device)
            self.agent_encoder = TransEncoder(context_dim=self.context_dim, n_blocks=cfg.n_blocks, n_heads=4,
                                              is_agent=True).to(ptu.device)
        elif self.encoder_type == "rnn":
            self.temporal_encoder = RNNEncoder(context_dim=self.context_dim, is_agent=False).to(ptu.device)
            self.agent_encoder = RNNEncoder(context_dim=self.context_dim, is_agent=True).to(ptu.device)
        elif self.encoder_type == "retnet":
            self.temporal_encoder = RetNetEncoder(context_dim=self.context_dim, n_blocks=3, n_heads=4,
                                                  is_agent=False).to(ptu.device)
            self.agent_encoder = RetNetEncoder(context_dim=self.context_dim, n_blocks=3, n_heads=4,
                                               is_agent=True).to(ptu.device)
        else:
            raise NotImplementedError("ae type " + cfg.ae_type + " not implemented")
        self.local_encoders = [
            LocalEncoder(
                context_dim=self.context_dim,
                hidden_sizes=cfg.hidden_sizes_mlp,
            ).to(ptu.device)
            for _ in range(self.n_agents)]
        self.global_log_std = Mlp(self.context_dim, self.context_dim, hidden_sizes=[]).to(ptu.device)
        self.local_log_std = Mlp(self.context_dim, self.context_dim, hidden_sizes=[]).to(ptu.device)

        encoder_parameters = []
        encoder_parameters.extend(self.temporal_encoder.parameters())
        encoder_parameters.extend(self.agent_encoder.parameters())
        for i in range(self.n_agents):
            encoder_parameters.extend(self.transition_encoders[i].parameters())
            encoder_parameters.extend(self.local_encoders[i].parameters())
        encoder_parameters.extend(self.global_log_std.parameters())
        encoder_parameters.extend(self.local_log_std.parameters())
        self.encoder_optimizer = AdamW(encoder_parameters, lr=self.lr)
        # endregion

    def transition_encode(self, obs_n, action_n, reward_n, next_obs_n):
        # (n_agents, meta_batch, batch_size, xxx_dim) x4 -> (meta_batch, batch_size, n_agents, context_dim)
        context_n = [ptu.zeros(0) for _ in range(self.n_agents)]
        for agent_id in range(self.n_agents):
            context_n[agent_id] = self.transition_encoders[agent_id](obs_n[agent_id], action_n[agent_id],
                                                                     reward_n[agent_id], next_obs_n[agent_id])
        return torch.stack(context_n, dim=-2)

    def temporal_encode(self, context_n):
        # (meta_batch, batch_size, n_agents, context_dim) -> (meta_batch, n_agents, context_dim)
        return self.temporal_encoder(context_n)

    def global_encode(self, context_ag):
        # (meta_batch, n_agents, context_dim) -> (meta_batch, context_dim)
        z = self.agent_encoder(context_ag)
        return z, self.global_log_std(z)

    def local_encode(self, context_ag):
        # (n_agents, meta_batch, batch_size, xxx_dim) -> (meta_batch, n_agents, context_dim)
        local_input = context_ag.detach().clone()
        z = [self.local_encoders[agent_id](local_input[:, agent_id, :]) for agent_id in range(self.n_agents)]
        z = torch.stack(z, dim=-2)
        return z, self.local_log_std(z)

    def forward(self, obs_n, action_n, reward_n, next_obs_n):
        context_n = self.transition_encode(obs_n, action_n, reward_n, next_obs_n)
        context_ag = self.temporal_encode(context_n)
        z_g, log_std_g = self.global_encode(context_ag)
        z_l, log_std_l = self.local_encode(context_ag)
        return z_g, log_std_g, z_l, log_std_l

    def encode(self, obs_n, action_n, reward_n, next_obs_n):
        z_g, log_std_g, z_l, log_std_l = self.forward(obs_n, action_n, reward_n, next_obs_n)
        return z_g + torch.randn_like(z_g) * F.softplus(log_std_g), \
               z_l + torch.randn_like(z_l) * F.softplus(log_std_l)

    # region update
    def update(self, data: Dict[str, torch.Tensor]):

        z_g, log_std_g, z_l, log_std_l = self.forward(*data["query"][:4])
        loss = self.kl_weight * (self.kl_div(z_g, log_std_g) + self.kl_div(z_l, log_std_l))
        z_g_q = z_g + torch.randn_like(z_g) * F.softplus(log_std_g)
        z_l_q = z_l + torch.randn_like(z_l) * F.softplus(log_std_l)
        update_info = {"kl_loss": loss.item()}

        key_pos, key_neg = data["key_pos"], data["key_neg"]
        key_pos = key_pos[:4]  # 剔除done
        for i in range(4):  # -> (n_agents, meta_batch * (n_tasks-1), batch_size, xxx_dim)
            key_neg[i] = key_neg[i].transpose(0, 1).reshape(self.n_agents, -1, *key_neg[i].shape[3:])
        z_g_p, _ = self.encode(*key_pos)
        z_g_n, z_l_n = self.encode(*key_neg)
        z_g_n = z_g_n.reshape(z_g_p.shape[0], self.n_tasks - 1, -1)

        global_loss, global_info = self.cl_loss(z_g_q, z_g_p, z_g_n)
        loss += global_loss
        update_info.update(global_info)
        local_loss, local_info = self.local_loss(z_g_q, z_g_p, z_g_n, z_l_q)
        loss += local_loss
        update_info.update(local_info)

        self.encoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        return update_info

    def cl_loss(self, z_q, z_p, z_n):
        # input_dim: (meta_batch, context_dim)
        b = z_q.shape[0]
        score_p = torch.bmm(z_q.view(b, 1, -1), z_p.view(b, -1, 1))  # (meta_batch, 1)
        score_n = torch.bmm(z_q.view(b, 1, -1), z_n.transpose(1, 2))  # (meta_batch, n_tasks - 1)
        y_hat = torch.cat([score_p.view(b, 1), score_n.view(b, self.n_tasks - 1)], dim=-1)  # (meta_batch, n_tasks)

        cl_loss = F.cross_entropy(y_hat, ptu.zeros(b, dtype=torch.long))
        cl_acc = (y_hat.argmax(dim=-1) == 0).float().mean()
        return cl_loss, {"cl_train_loss": cl_loss.item(), "cl_train_acc": cl_acc}

    def local_loss(self, _z_g_q, _z_g_p, _z_g_n, z_l_q):
        z_g_q, z_g_p, z_g_n = _z_g_q.detach().clone(), _z_g_p.detach().clone(), _z_g_n.detach().clone()
        b = z_g_q.shape[0]
        score_g_p = torch.bmm(z_g_q.view(b, 1, -1), z_g_p.view(b, -1, 1)).repeat(1, self.n_agents, 1)
        score_l_p = torch.bmm(z_l_q, z_g_p.unsqueeze(-1).repeat(1, 1, self.n_agents))
        score_l_p = score_l_p[:, torch.arange(self.n_agents), torch.arange(self.n_agents)].view(b, self.n_agents, 1)
        # (meta_batch, n_agents, 1)
        loss1 = (score_l_p - score_g_p).mean()

        score_g_n = torch.bmm(z_g_q.view(b, 1, -1), z_g_n.transpose(1, 2)).repeat(1, self.n_agents, 1)
        score_l_n = torch.bmm(z_l_q.view(b * self.n_agents, 1, -1),
                              z_g_n.unsqueeze(1).repeat(1, self.n_agents, 1, 1).transpose(2, 3).view(b * self.n_agents, self.context_dim, -1))
        score_l_n = score_l_n.view(b, self.n_agents, self.n_tasks - 1)

        score_g_all = torch.cat([score_g_p, score_g_n], dim=-1)
        num = torch.exp(score_g_all).sum(dim=-1)
        den = torch.exp(score_l_n).sum(dim=-1)
        loss2 = torch.log(num / den).mean()

        loss = loss1 + loss2

        return loss, {"local_loss": loss.item(), "local_loss1": loss1.item(), "local_loss2": loss2.item()}

    # endregion

    def lr_decay(self, epoch, total_epoch):
        update_linear_schedule(self.encoder_optimizer, epoch, total_epoch, self.lr)

    def save_model(self, save_path):
        models = {
            "transition_encoders": self.transition_encoders,
            "temporal_encoder": self.temporal_encoder,
            "agent_encoder": self.agent_encoder,
            "local_encoders": self.local_encoders,
        }
        save_path = os.path.join(save_path, "ae.pth")
        with open(save_path, "wb") as f:
            torch.save(models, f)

    def load_model(self, load_path):
        load_path = os.path.join(load_path, "ae.pth")
        with open(load_path, "rb") as f:
            models = torch.load(f, map_location=ptu.device)
        self.transition_encoders = [encoder.to(ptu.device) for encoder in models["transition_encoders"]]
        self.temporal_encoder = models["temporal_encoder"].to(ptu.device)
        self.agent_encoder = models["agent_encoder"].to(ptu.device)
        self.local_encoders = [encoder.to(ptu.device) for encoder in models["local_encoders"]]
