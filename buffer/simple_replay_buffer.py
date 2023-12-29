from typing import Dict, List, Tuple, Any
import numpy as np


# 单智能体单任务
# 每幕结束才会压入, 不存在episode没结束就压入的情况
class SimpleReplayBuffer:
    def __init__(
            self,
            max_size,
            obs_dim,
            action_dim,
            max_episode_steps,
            **kwargs,
    ):
        self.max_size = max_size
        self.observation_dim = obs_dim
        self.action_dim = action_dim
        self.max_episode_steps = max_episode_steps

        self._obs = np.zeros((max_size, obs_dim))
        self._actions = np.zeros((max_size, action_dim))
        self._reward = np.zeros((max_size, 1))
        self._next_obs = np.zeros((max_size, obs_dim))
        self._done = np.zeros((max_size, 1))

        self._top = 0
        self.size = 0
        # self._episode_starts = []
        self.episode_indices = []
        # List[List[int]], 每个episode的index, 改进后可以用于每幕长度不一的情况

    def clear(self):
        self._top = 0
        self.size = 0
        # self._episode_starts = []
        self.episode_indices = []

    def add_sample(self, obs, action, reward, next_obs, done):
        self._obs[self._top] = obs
        self._actions[self._top] = action
        self._reward[self._top] = reward
        self._next_obs[self._top] = next_obs
        self._done[self._top] = done

        self._top = (self._top + 1) % self.max_size
        if self.size < self.max_size:
            self.size += 1

    def add_samples(self, n_obs, n_action, n_reward, n_next_obs, n_done, **kwargs):
        # np.ndarray (n_samples, xxx_dim)
        n_samples = n_obs.shape[0]
        if self._top + n_samples <= self.max_size:
            self._obs[self._top: self._top + n_samples] = n_obs
            self._actions[self._top: self._top + n_samples] = n_action
            self._reward[self._top: self._top + n_samples] = n_reward
            self._next_obs[self._top: self._top + n_samples] = n_next_obs
            self._done[self._top: self._top + n_samples] = n_done

            self._top = (self._top + n_samples) % self.max_size
            self.size = min(self.size + n_samples, self.max_size)
        else:
            for i in range(n_samples):
                self.add_sample(n_obs[i], n_action[i], n_reward[i], n_next_obs[i], n_done[i])

    def add_episode(self, ep_obs, ep_action, ep_reward, ep_next_obs, ep_done):
        # np.ndarray (episode_len, xxx_dim)
        if self._top + ep_obs.shape[0] <= self.max_size:
            self.episode_indices.append(list(range(self._top, self._top + ep_obs.shape[0])))
            self.add_samples(ep_obs, ep_action, ep_reward, ep_next_obs, ep_done)
        else:
            self.episode_indices.append([])
            for i in range(ep_obs.shape[0]):
                self.episode_indices[-1].append(self._top)
                self.add_sample(ep_obs[i], ep_action[i], ep_reward[i], ep_next_obs[i], ep_done[i])
        if self.size == self.max_size:
            while set(self.episode_indices[0]) & set(self.episode_indices[-1]):
                del self.episode_indices[0]

    def sample_data(self, indices):
        return dict(
            obs=self._obs[indices],
            action=self._actions[indices],
            reward=self._reward[indices],
            next_obs=self._next_obs[indices],
            done=self._done[indices],
        )

    def random_batch(self, batch_size):
        indices = np.random.randint(0, self.size, batch_size)
        return self.sample_data(indices)

    def random_sequence(self, batch_size):
        """依旧采样batch_size个样本, 但这些样本是连续的"""
        indices = []
        while len(indices) < batch_size:
            start = np.random.randint(low=0, high=len(self.episode_indices))
            indices += self.episode_indices[start]
        indices = indices[:batch_size]
        return self.sample_data(indices)

    def random_episodes(self, n_episodes):
        """随机采样n_episodes个episode, 返回List[Dict]"""
        sampled_data = []
        for ep in range(n_episodes):
            start = np.random.randint(low=0, high=len(self.episode_indices))
            sampled_data.append(self.sample_data(self.episode_indices[start]))
        return sampled_data

    # 一些原版提供了的样板功能型函数
    def can_sample_batch(self, batch_size):
        return batch_size <= self.size

    def can_sample_episodes(self, n_episodes=0):
        return len(self.episode_indices) >= n_episodes

    def num_steps_can_sample(self):
        return self.size

    def num_complete_episodes(self):
        return len(self.episode_indices)














