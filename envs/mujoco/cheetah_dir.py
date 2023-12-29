"""adapted from https:github.com/cnfinn/maml_rl/rllab/envs/mujoco"""
import numpy as np

from envs.mujoco.multiagent_mujoco.mujoco_multi import MujocoMulti


class HalfCheetahDirEnvMulti(MujocoMulti):
    def __init__(self, n_agents, n_tasks=2, max_episode_steps=100, task=None, success_factor=0.0, **kwargs):
        if "agent_obsk" not in kwargs:
            kwargs["agent_obsk"] = 1
        if n_agents == 2:
            env_args = {"scenario": "HalfCheetah-v2",
                        "agent_conf": "2x3",
                        "agent_obsk": kwargs["agent_obsk"],
                        "episode_limit": max_episode_steps}
        elif n_agents == 6:
            env_args = {"scenario": "HalfCheetah-v2",
                        "agent_conf": "6x1",
                        "agent_obsk": kwargs["agent_obsk"],
                        "episode_limit": max_episode_steps}
        else:
            raise NotImplementedError("num of agents: %d doesn't match HalfCheetah" % n_agents)

        task = {"goal": 1.} if task is None else task
        self._task = task
        self._goal = task["goal"]
        self.num_tasks = n_tasks
        self.tasks = self.sample_tasks(n_tasks)
        self.max_episode_steps = max_episode_steps
        self.velocity = 0.
        self.success_factor = success_factor

        super(HalfCheetahDirEnvMulti, self).__init__(env_args=env_args)
        self.reset_task(0)

    def revise_reward(self, info):
        return self._task["goal"] * info['reward_run'] + info['reward_ctrl'] \
               + self.success_factor * info["is_goal_state"]

    def step(self, actions):
        reward, done, info = super().step(actions)
        self.velocity = info["reward_run"]
        info["is_goal_state"] = True if info["reward_run"] * self._goal > 0 else False
        info["goal"] = self.get_goal()
        obs_n = self.get_obs()

        reward = self.revise_reward(info)
        reward_n = [np.array(reward) for _ in range(self.n_agents)]
        done_n = [done for _ in range(self.n_agents)]
        return obs_n, reward_n, done_n, info

    def sample_tasks(self, n_tasks):
        # -1后退, +1前进
        return [{"goal": -1.0}, {"goal": 1.0}]

    def set_task(self, task):
        self._task = task

    def get_task(self):
        return self._task

    def set_goal(self, goal):
        self._goal = goal
        self._task["goal"] = goal

    def get_goal(self):
        return np.array(self._goal)

    def reset_task(self, task_idx=None):
        if task_idx is None:
            task_idx = np.random.randint(self.num_tasks)
        self._task = self.tasks[task_idx]
        self._goal = self._task["goal"]
        # self.reset()
        self.velocity = 0.

    def get_all_task_idx(self):
        return list(range(self.num_tasks))


class HalfCheetahDirOracleEnvMulti(HalfCheetahDirEnvMulti):
    def get_obs_agent(self, agent_id):
        task_obs = [0., 1.] if self._task["goal"] < 0 else [1., 0.]
        obs = super().get_obs_agent(agent_id)
        return np.concatenate([task_obs, obs])








