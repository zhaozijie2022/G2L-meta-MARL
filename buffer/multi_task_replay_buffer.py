import numpy as np
from typing import Dict, List, Tuple, Any
from buffer.simple_replay_buffer import SimpleReplayBuffer
from buffer.multi_agent_replay_buffer import MultiAgentReplayBuffer
import pickle


# 多智能体多任务, Dict[task_idx, MultiAgentReplayBuffer]
class MultiTaskReplayBuffer:
    def __init__(
            self,
            max_size,
            obs_dim_n: List[int],
            action_dim_n: List[int],
            max_episode_steps,
            tasks: List[int],
            n_agents,
            **kwargs,
    ):
        self.max_size = max_size
        self.obs_dim_n = obs_dim_n
        self.action_dim_n = action_dim_n
        self.max_episode_steps = max_episode_steps
        self.tasks = tasks
        self.n_agents = n_agents
        self.task_buffers = dict([(idx, MultiAgentReplayBuffer(
            max_size=max_size,
            obs_dim_n=obs_dim_n,
            action_dim_n=action_dim_n,
            max_episode_steps=max_episode_steps,
            n_agents=n_agents,
            **kwargs
        )) for idx in self.tasks])
        # 由于不同任务的采样是异步的, 所以不需要top, size等内存管理

    def clear(self):
        for task_idx in self.tasks:
            self.task_buffers[task_idx].clear()

    def add_sample(self, task_idx, obs_n: List[np.ndarray],
                   action_n, reward_n, next_obs_n, done_n,):
        self.task_buffers[task_idx].add_sample(obs_n=obs_n, action_n=action_n, reward_n=reward_n,
                                               next_obs_n=next_obs_n, done_n=done_n)

    def add_samples(self, task_idx, n_obs_n: List[List[np.ndarray]],
                    n_action_n, n_reward_n, n_next_obs_n, n_done_n):
        self.task_buffers[task_idx].add_samples(n_obs_n=n_obs_n, n_action_n=n_action_n, n_reward_n=n_reward_n,
                                                n_next_obs_n=n_next_obs_n, n_done_n=n_done_n)

    def add_episode(self, task_idx, ep_obs_n: List[List[np.ndarray]],
                    ep_action_n, ep_reward_n, ep_next_obs_n, ep_done_n):
        """List[List[np.ndarray(xxx_dim,), n_agents], n_steps]"""
        self.task_buffers[task_idx].add_episode(ep_obs_n=ep_obs_n, ep_action_n=ep_action_n, ep_reward_n=ep_reward_n,
                                                ep_next_obs_n=ep_next_obs_n, ep_done_n=ep_done_n)

    def sample_data(self, task_idx, indices):
        return self.task_buffers[task_idx].sample_data(indices)

    def random_batch(self,task_idx, batch_size, sequence=False):
        # return List[Dict[str, np.ndarray], n_agents]
        if sequence:
            batch = self.task_buffers[task_idx].random_sequence(batch_size)
        else:
            batch = self.task_buffers[task_idx].random_batch(batch_size)
        return batch

    def random_episodes(self, task_idx, n_episodes):
        return self.task_buffers[task_idx].random_episodes(n_episodes)

    # 功能函数
    def can_sample_batch(self, task_idx, batch_size):
        return self.task_buffers[task_idx].can_sample_batch(batch_size)

    def can_sample_episodes(self, task_idx, n_episodes):
        return self.task_buffers[task_idx].can_sample_episodes(n_episodes)

    def num_steps_can_sample(self, task_idx):
        return self.task_buffers[task_idx].num_steps_can_sample()

    def add_path(self, task_idx, path):
        # TODO
        # self.task_buffers[task_idx].add_path(path)
        pass

    def add_paths(self, task_idx, paths):
        # TODO
        # self.task_buffers[task_idx].add_paths(paths)
        pass

    def clear_buffer(self, task_idx):
        self.task_buffers[task_idx].clear()

    def num_complete_episodes(self, task_idx):
        return self.task_buffers[task_idx].num_complete_episodes()

    def save_buffer(self, task_idx, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.task_buffers[task_idx], f)









































