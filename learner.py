import datetime
import json
import os
import time
import copy
from argparse import Namespace
from itertools import chain
from typing import List, Dict

import gym
import numpy as np
import torch
from gym.spaces import Box
from omegaconf import DictConfig, OmegaConf
from collections import Counter

from ae.make_ae import make_ae
from algos.make_algo import make_algo
from buffer.make_buffer import make_buffer
from envs.make_env import make_env
from torchkit import pytorch_utils as ptu
from utils import helpers as utl


class Learner:
    def __init__(self, cfg: DictConfig):
        self.cfg = Namespace(**OmegaConf.to_container(cfg, resolve=True))
        utl.seed(self.cfg.seed)

        self.envs = make_env(cfg=self.cfg)
        print("initial envs: %s, done" % cfg.save_name)
        self.envs.reset_task(0)
        self.n_agents = self.cfg.n_agents
        self.n_tasks = self.cfg.n_tasks
        self.max_ep_len = self.cfg.max_ep_len
        self.obs_dim_n = [self.envs.observation_space[i].shape[0] for i in range(self.n_agents)]
        self.action_dim_n = [self.envs.action_space[i].n if isinstance(self.envs.action_space[i], gym.spaces.Discrete)
                             else self.envs.action_space[i].shape[0] for i in range(self.n_agents)]
        self.cfg.obs_dim_n = self.obs_dim_n
        self.cfg.action_dim_n = self.action_dim_n
        self.context_dim = self.cfg.context_dim
        self.task_idxes = list(range(self.cfg.n_tasks))
        try:
            self.cfg.state_dim = self.envs.state_dim
        except AttributeError:
            self.cfg.state_dim = sum(self.cfg.obs_dim_n)

        self.use_centralized_V = self.cfg.use_centralized_V
        self.algo_hidden_size = self.cfg.algo_hidden_size
        self.recurrent_N = self.cfg.recurrent_N
        self.cfg.observation_space = self.envs.observation_space[:]
        self.cfg.action_space = self.envs.action_space[:]
        if self.use_centralized_V:
            self.cfg.share_observation_space = [
                Box(low=-np.inf, high=+np.inf, shape=(sum(self.cfg.obs_dim_n),), dtype=np.float32)
                for _ in range(self.cfg.n_agents)]
        self.agents = make_algo(cfg=self.cfg)
        print("initial algorithm: %s, done" % cfg.algo_file)

        self.ae = make_ae(cfg=self.cfg)
        print("initial context ae, done")

        self.ae_buffer_size = self.cfg.ae_buffer_size
        self.sample_ae_interval = self.cfg.sample_ae_interval
        self.rl_buffer, self.ae_buffer = make_buffer(self.cfg)  # for sample rl batch
        tmp_cfg = copy.deepcopy(self.cfg)
        tmp_cfg.n_rollout_threads = self.cfg.n_ae_rollout_threads
        self.dummy_envs = make_env(tmp_cfg)
        self.dummy_buffer, _ = make_buffer(tmp_cfg)  # dummy_x for sample context batch
        print("initial rl/ae buffer, done")

        self.use_linear_lr_decay = self.cfg.use_linear_lr_decay
        self.n_iters = self.cfg.n_iters
        self.n_init_rollouts = self.cfg.n_init_rollouts
        self.n_rollout_threads = self.cfg.n_rollout_threads
        self.n_ae_rollout_threads = self.cfg.n_ae_rollout_threads
        self.n_eval_rollout_threads = self.cfg.n_eval_rollout_threads
        self.ae_batch_size = self.cfg.ae_batch_size
        self.ae_updates_per_iter = self.cfg.ae_updates_per_iter
        self.eval_interval = self.cfg.eval_interval

        self.is_save_model = self.cfg.save_model
        self.save_interval = self.cfg.save_interval
        self.log_interval = self.cfg.log_interval
        env_dir = self.cfg.save_name

        if self.cfg.load_model:
            self.load_model(self.cfg.load_model_path)
            print("Note: Load model, done!!!!")

        date_dir = datetime.datetime.now().strftime("%m%d_%H%M_")
        seed_dir = 'seed{}'.format(self.cfg.seed)
        self.expt_name = date_dir + seed_dir
        if self.is_save_model:
            os.makedirs(cfg.main_save_path, exist_ok=True)
            self.output_path = os.path.join(self.cfg.main_save_path, env_dir, self.expt_name)
            self.cfg.output_path = self.output_path
            os.makedirs(self.output_path, exist_ok=True)
            with open(os.path.join(self.output_path, 'config.json'), 'w') as f:
                config_json = vars(self.cfg)
                config_json.pop("action_space")
                config_json.pop("observation_space")
                config_json.pop("share_observation_space")
                json.dump(config_json, f, indent=4)

        print("initial learner, done")
        self._start_time = time.time()
        self._check_time = time.time()

    def train(self):
        self._check_time = time.time()
        for _ in range(self.n_init_rollouts):
            ae_meta_tasks = np.random.choice(self.task_idxes, self.n_ae_rollout_threads, replace=True)
            self.rollout(ae_meta_tasks, self.dummy_buffer, self.dummy_envs, True)
        print("Initial ae buffer, done. time: %.2fs" % (time.time() - self._check_time))

        self._check_time = time.time()
        for iter_ in range(self.n_iters):
            if self.use_linear_lr_decay:
                for i in range(self.n_agents):
                    self.agents[i].policy.lr_decay(iter_, self.n_iters)
                self.ae.lr_decay(iter_, self.n_iters)

            meta_tasks = np.random.choice(self.task_idxes, self.n_rollout_threads, replace=True)
            _rew, _sr, _rew_p = self.rollout(meta_tasks, self.rl_buffer, self.envs, False)
            task_env_nums = Counter(meta_tasks)
            rew_tasks = np.zeros(self.n_tasks) + np.random.rand(self.n_tasks) * 1e-5
            for i_t, task_idx in enumerate(meta_tasks):
                rew_tasks[task_idx] += _rew_p[i_t, 0] / task_env_nums[task_idx]
            rollout_info = {"reward": np.mean(rew_tasks), "rl-collect-sr": _sr, }
            for task_idx in self.task_idxes:
                rollout_info["reward-%d" % task_idx] = rew_tasks[task_idx]

            # update rl  < note here back Q Loss to ae>
            rl_train_info = self.rl_update(meta_tasks)

            ae_train_info = self.ae_update(meta_tasks=meta_tasks)

            if (iter_ + 1) % 1 == 0:
                self.log(iter_ + 1, meta_tasks, rollout_info=rollout_info,
                         ae_train_info=ae_train_info, rl_train_info=rl_train_info)
                print([self.ae_buffer.task_buffers[task_idx].size for task_idx in self.task_idxes])

            if self.is_save_model and (iter_ + 1) % self.save_interval == 0:
                save_path = os.path.join(self.output_path, 'models_%d' % (iter_ + 1))
                if self.is_save_model:
                    os.makedirs(save_path, exist_ok=True)
                    self.save_model(save_path)
                    print("model saved in %s" % save_path)

            if (iter_ + 1) % self.sample_ae_interval == 0:
                ae_meta_tasks = np.random.choice(self.task_idxes, self.n_ae_rollout_threads, replace=True)
                self.rollout(ae_meta_tasks, self.dummy_buffer, self.dummy_envs, True)

        self.envs.close()
        self.dummy_envs.close()
        print("multi processing envs have been closed")
        print("")

    def rollout(self, meta_tasks, r_buffer, r_envs, is_ae_buffer=False):
        meta_batch = len(meta_tasks)
        assert r_envs.n_envs == meta_batch
        assert r_buffer[0].n_rollout_threads == meta_batch

        _rew, _sr = 0., 0.
        _rew_p = np.zeros((meta_batch, 1))
        self.warmup(meta_tasks, r_buffer, r_envs)
        _done = np.zeros((self.n_agents, self.max_ep_len, meta_batch), dtype=np.float32)
        _actions_ae = np.zeros((self.n_agents, self.max_ep_len, meta_batch, self.action_dim_n[0]), dtype=np.float32)

        for cur_step in range(self.max_ep_len):
            with torch.no_grad():
                if not is_ae_buffer:
                    data4z = self.sample_batch_ae(meta_tasks, self.ae_batch_size)
                    _, z = self.ae.encode(*data4z)
                    z = z.permute(1, 0, 2)
                else:
                    z = ptu.randn(self.n_agents, self.n_ae_rollout_threads, self.context_dim)

            (values, actions, action_log_probs, rnn_states,
             rnn_states_critic, actions_ae) = self.collect(cur_step, z, r_buffer)
            obs, rewards, dones, infos = r_envs.step(actions)
            data = (obs, rewards, dones, infos, values, actions,
                    action_log_probs, rnn_states, rnn_states_critic,)
            self.insert(data, r_buffer)

            _done[:, cur_step, :] = np.transpose(dones, (1, 0)).copy()
            _actions_ae[:, cur_step, :, :] = np.transpose(actions_ae, (1, 0, 2)).copy()
            _rew += np.mean(rewards)
            _sr += (np.mean([info["is_goal_state"] for info in infos]) / self.max_ep_len)
            _rew_p += np.mean(rewards, axis=1)

        if is_ae_buffer:
            for i_t, task_idx in enumerate(meta_tasks):
                self.ae_buffer.add_samples(
                    task_idx=task_idx,
                    n_obs_n=[r_buffer[agent_id].obs[:-1, i_t, :] for agent_id in range(self.n_agents)],
                    n_action_n=_actions_ae[:, :, i_t, :],
                    n_reward_n=[r_buffer[agent_id].rewards[:, i_t, :] for agent_id in range(self.n_agents)],
                    n_next_obs_n=[r_buffer[agent_id].obs[1:, i_t, :] for agent_id in range(self.n_agents)],
                    n_done_n=_done[:, :, i_t],
                )
        else:
            data4z = self.sample_batch_ae(meta_tasks, self.ae_batch_size)
            z_g, _ = self.ae.encode(*data4z)
            self.compute(r_buffer, z_g)
        return _rew, _sr, _rew_p

    def warmup(self, meta_tasks, r_buffer, r_envs):
        r_envs.meta_reset_task(meta_tasks)
        obs = r_envs.reset()
        share_obs = obs.reshape(obs.shape[0], -1).copy()

        for agent_id in range(self.n_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            r_buffer[agent_id].share_obs[0] = share_obs.copy()
            r_buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, cur_step, z, r_buffer):
        values = []
        actions = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.n_agents):
            self.agents[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic = \
                self.agents[agent_id].policy.get_actions(
                    cent_obs=r_buffer[agent_id].share_obs[cur_step],
                    obs=r_buffer[agent_id].obs[cur_step],
                    rnn_states_actor=r_buffer[agent_id].rnn_states[cur_step],
                    rnn_states_critic=r_buffer[agent_id].rnn_states_critic[cur_step],
                    masks=r_buffer[agent_id].masks[cur_step],
                    context=z[agent_id], )
            values.append(ptu.get_numpy(value))
            action = ptu.get_numpy(action)
            actions.append(action)
            action_log_probs.append(ptu.get_numpy(action_log_prob))
            rnn_states.append(ptu.get_numpy(rnn_state))
            rnn_states_critic.append(ptu.get_numpy(rnn_state_critic))

        values = np.array(values).transpose((1, 0, 2))
        actions = np.array(actions).transpose((1, 0, 2))
        if self.envs.action_space[0].__class__.__name__ == "Discrete":
            actions_ae = np.eye(self.envs.action_space[0].n)[actions.reshape(-1)].reshape(*actions.shape[:2], -1)
        else:
            actions_ae = actions.copy()
        action_log_probs = np.array(action_log_probs).transpose((1, 0, 2))
        rnn_states = np.array(rnn_states).transpose((1, 0, 2, 3))
        rnn_states_critic = np.array(rnn_states_critic).transpose((1, 0, 2, 3))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_ae

    def insert(self, data, r_buffer):
        (obs, rewards, dones, infos, values, actions,
         action_log_probs, rnn_states, rnn_states_critic,) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.algo_hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.algo_hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((r_buffer[0].n_rollout_threads, self.n_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.n_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            r_buffer[agent_id].insert(
                share_obs,
                np.array(list(obs[:, agent_id])),
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],
            )

    @torch.no_grad()
    def compute(self, r_buffer, z):
        for agent_id in range(self.n_agents):
            self.agents[agent_id].prep_rollout()
            next_value = self.agents[agent_id].policy.get_values(
                cent_obs=r_buffer[agent_id].share_obs[-1],
                context=z,
                rnn_states_critic=r_buffer[agent_id].rnn_states_critic[-1],
                masks=r_buffer[agent_id].masks[-1],
            )
            next_value = ptu.get_numpy(next_value)
            r_buffer[agent_id].compute_returns(next_value, self.agents[agent_id].value_normalizer)

    def sample_batch_ae(self, meta_tasks, batch_size):
        batch = []
        for task_idx in meta_tasks:
            batch.append(self.ae_buffer.random_batch(task_idx=task_idx, batch_size=batch_size))
        obs_n = torch.stack([torch.stack([ptu.from_numpy(tmp2["obs"]) for tmp2 in tmp1]) for tmp1 in batch], dim=1)
        action_n = torch.stack([torch.stack([ptu.from_numpy(tmp2["action"]) for tmp2 in tmp1]) for tmp1 in batch],
                               dim=1)
        reward_n = torch.stack([torch.stack([ptu.from_numpy(tmp2["reward"]) for tmp2 in tmp1]) for tmp1 in batch],
                               dim=1)
        next_obs_n = torch.stack([torch.stack([ptu.from_numpy(tmp2["next_obs"]) for tmp2 in tmp1]) for tmp1 in batch],
                                 dim=1)
        return [obs_n, action_n, reward_n, next_obs_n]


    def rl_update(self, meta_tasks):
        data4z = self.sample_batch_ae(meta_tasks, self.ae_batch_size)

        update_info = {}
        count = 0
        for agent_id in range(self.n_agents):
            self.agents[agent_id].prep_training()
            agent_train_info = self.agents[agent_id].train(
                buffer=self.rl_buffer[agent_id],
                update_actor=True,
                ae=self.ae,
                data4z=data4z,
            )
            # we embed backward Q Loss into the MARL update
            self.rl_buffer[agent_id].after_update()

            count += 1
            for key, value in agent_train_info.items():
                if key not in update_info:
                    update_info[key] = value
                else:
                    update_info[key] += value
        for key, value in update_info.items():
            update_info[key] = value / count
        return update_info

    def ae_update(self, meta_tasks):

        meta_batch = len(meta_tasks)
        update_info = {}
        for r in range(self.ae_updates_per_iter):
            data = {"query": self.sample_batch_ae(meta_tasks, self.ae_batch_size),
                    "key_pos": self.sample_batch_ae(meta_tasks, self.ae_batch_size)}
            obs_neg, action_neg, reward_neg, next_obs_neg = [[[] for _ in meta_tasks] for _ in range(4)]
            for i in range(meta_batch):
                neg_meta_batch = self.task_idxes[:meta_tasks[i]] + self.task_idxes[meta_tasks[i] + 1:]
                obs_neg[i], action_neg[i], reward_neg[i], next_obs_neg[i] = \
                    self.sample_batch_ae(neg_meta_batch, self.ae_batch_size)
            data["key_neg"] = [torch.stack(obs_neg), torch.stack(action_neg),
                               torch.stack(reward_neg), torch.stack(next_obs_neg)]

            ae_losses = self.ae.update(data)
            for key, value in ae_losses.items():
                if key in update_info:
                    update_info[key] += value
                else:
                    update_info[key] = value
            if update_info["cl_train_acc"] > 0.99 * (r + 1):
                update_info["cl_train_times"] = r + 1
                break
        if "cl_train_times" not in update_info:
            for key, value in update_info.items():
                update_info[key] = value / self.ae_updates_per_iter
            update_info["cl_train_times"] = self.ae_updates_per_iter
        else:
            for key, value in update_info.items():
                update_info[key] = value / update_info["cl_train_times"]
        return update_info

    def log(self, iter_, meta_tasks, **kwargs):

        print("")
        print("******** iter: %d, iter_time: %.2fs, total_time: %.2fs" %
              (iter_, time.time() - self._check_time, time.time() - self._start_time))
        print("meta_tasks: ", meta_tasks)
        for key, value in kwargs.items():
            print("%s" % key + "".join([", %s: %.4f" % (k, v) for k, v in value.items()]))
        self._check_time = time.time()

    def save_model(self, save_path):
        for agent_id in range(self.n_agents):
            self.agents[agent_id].save_model(save_path)
        self.ae.save_model(save_path)

    def load_model(self, load_path):
        for agent_id in range(self.n_agents):
            self.agents[agent_id].load_model(load_path)
        self.ae.load_model(load_path)
