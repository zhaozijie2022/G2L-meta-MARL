import numpy as np
from buffer.simple_replay_buffer import SimpleReplayBuffer
from typing import Dict, List, Tuple, Any
import pickle


# 多智能体单任务, Dict[agent_id, SimpleReplayBuffer]
class MultiAgentReplayBuffer:
    def __init__(
            self,
            max_size,
            obs_dim_n: List[int],
            action_dim_n: List[int],
            max_episode_steps,
            n_agents,
            **kwargs,
    ):

        self.max_size = max_size
        self.obs_dim_n = obs_dim_n
        self.action_dim_n = action_dim_n
        self.max_episode_steps = max_episode_steps
        self.n_agents = n_agents
        self.agent_buffers = dict([(idx, SimpleReplayBuffer(
            max_size=max_size,
            obs_dim=obs_dim_n[idx],
            action_dim=action_dim_n[idx],
            max_episode_steps=max_episode_steps,
            **kwargs
        )) for idx in range(self.n_agents)])

        self._top = 0
        self.size = 0
        self.episode_indices = []
        # 所有agent的transition同步, 在SimpleReplayBuffer中依旧进行了内存管理, 牺牲速度提升可读性

    def clear(self):
        self._top = 0
        self.size = 0
        self.episode_indices = []
        for agent_id in range(self.n_agents):
            self.agent_buffers[agent_id].clear()

    def add_sample(self, obs_n: List[np.ndarray],
                   action_n, reward_n, next_obs_n, done_n,):
        """List[np.ndarray(xxx_dim,), n_agents]"""
        for agent_id in range(self.n_agents):
            self.agent_buffers[agent_id].add_sample(obs=obs_n[agent_id],
                                                    action=action_n[agent_id],
                                                    reward=reward_n[agent_id],
                                                    next_obs=next_obs_n[agent_id],
                                                    done=done_n[agent_id])
        self._top = (self._top + 1) % self.max_size
        if self.size < self.max_size:
            self.size += 1

    def add_samples(self, n_obs_n: List[List[np.ndarray]],
                    n_action_n, n_reward_n, n_next_obs_n, n_done_n,):
        """List[List[np.ndarray(xxx_dim,), n_samples], n_agents]"""
        n_samples = len(n_obs_n[0])

        # -> List[np.ndarray(n_samples, xxx_dim), n_agents]
        n_obs_n_ = [np.array(n_obs_n[agent_id]).reshape(n_samples, -1) for agent_id in range(self.n_agents)]
        n_action_n_ = [np.array(n_action_n[agent_id]).reshape(n_samples, -1) for agent_id in range(self.n_agents)]
        n_reward_n_ = [np.array(n_reward_n[agent_id]).reshape(n_samples, -1) for agent_id in range(self.n_agents)]
        n_next_obs_n_ = [np.array(n_next_obs_n[agent_id]).reshape(n_samples, -1) for agent_id in range(self.n_agents)]
        n_done_n_ = [np.array(n_done_n[agent_id]).reshape(n_samples, -1) for agent_id in range(self.n_agents)]

        for agent_id in range(self.n_agents):
            self.agent_buffers[agent_id].add_samples(n_obs=n_obs_n_[agent_id],
                                                     n_action=n_action_n_[agent_id],
                                                     n_reward=n_reward_n_[agent_id],
                                                     n_next_obs=n_next_obs_n_[agent_id],
                                                     n_done=n_done_n_[agent_id],)

        self._top = (self._top + n_samples) % self.max_size
        self.size = self.agent_buffers[0].size

    def add_episode(self, ep_obs_n, ep_action_n, ep_reward_n, ep_next_obs_n, ep_done_n):
        """List[List[np.ndarray(xxx_dim, ), n_steps], n_agents]"""
        n_samples = len(ep_obs_n[0])
        if self._top + n_samples <= self.max_size:
            self.episode_indices.append(list(range(self._top, self._top + n_samples)))
            self.add_samples(ep_obs_n, ep_action_n, ep_reward_n, ep_next_obs_n, ep_done_n)
        else:
            self.episode_indices.append([])
            for i in range(n_samples):  # 每一时刻, trans_obs_n: List[np.ndarray(xxx_dim,), n_agents]
                self.episode_indices[-1].append(self._top)
                trans_obs_n = [ep_obs_n[i][agent_id] for agent_id in range(self.n_agents)]
                trans_action_n = [ep_action_n[i][agent_id] for agent_id in range(self.n_agents)]
                trans_reward_n = [ep_reward_n[i][agent_id] for agent_id in range(self.n_agents)]
                trans_next_obs_n = [ep_next_obs_n[i][agent_id] for agent_id in range(self.n_agents)]
                trans_done_n = [ep_done_n[i][agent_id] for agent_id in range(self.n_agents)]
                self.add_sample(trans_obs_n, trans_action_n, trans_reward_n, trans_next_obs_n, trans_done_n)

        if self.size == self.max_size:
            while set(self.episode_indices[0]) == set(range(self.max_episode_steps)):
                del self.episode_indices[0]

    def sample_data(self, indices):
        """return List[Dict[str, np.ndarray], n_agents]"""
        return [self.agent_buffers[agent_id].sample_data(indices)
                for agent_id in range(self.n_agents)]

    def random_batch(self, batch_size):
        """return List[Dict[str, np.ndarray], n_agents]"""
        indices = np.random.randint(0, self.size, size=batch_size)
        return self.sample_data(indices)

    def random_sequence(self, batch_size):
        """return List[Dict[str, np.ndarray], n_agents]"""
        indices = []
        while len(indices) < batch_size:
            start = np.random.randint(low=0, high=len(self.episode_indices))
            indices += self.episode_indices[start]
        indices = indices[:batch_size]
        return self.sample_data(indices)

    def random_episodes(self, n_episodes):
        """return List[List[Dict[str, np.ndarray], n_agents], n_episodes]"""
        sampled_data = []
        for ep in range(n_episodes):
            start = np.random.randint(low=0, high=len(self.episode_indices))
            sampled_data.append(self.sample_data(self.episode_indices[start]))
        return sampled_data

    # 功能函数
    def can_sample_batch(self, batch_size):
        return batch_size <= self.size

    def can_sample_episodes(self, n_episodes=0):
        return len(self.episode_indices) >= n_episodes

    def num_steps_can_sample(self):
        return self.size

    def num_complete_episodes(self):
        return len(self.episode_indices)











