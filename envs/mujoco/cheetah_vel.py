"""adapted from https:github.com/cnfinn/maml_rl/rllab/envs/mujoco"""
import numpy as np
from envs.mujoco.multiagent_mujoco.mujoco_multi import MujocoMulti


class HalfCheetahVelEnvMulti(MujocoMulti):
    def __init__(self, n_agents, n_tasks=3, max_episode_steps=1000, task={"velocity": 0.}, **kwargs):
        if n_agents == 2:
            env_args = {"scenario": "HalfCheetah-v2",
                        "agent_conf": "2x3",
                        "agent_obsk": 1,
                        "episode_limit": max_episode_steps,}
        elif n_agents == 6:
            env_args = {"scenario": "HalfCheetah-v2",
                        "agent_conf": "6x1",
                        "agent_obsk": 1,
                        "episode_limit": max_episode_steps,}
        else:
            raise NotImplementedError("num of agents: %d doesn't match HalfCheetah" % n_agents)

        # self.wrapped_env = gym.make("HalfCheetah-v2")
        # 在mujoco_multi中, self.env = self.wrapped_env.env.env, 类型是gym.envs.mujoco.half_cheetah.HalfCheetahEnv
        self._task = task
        self._goal = task["velocity"]
        self.num_tasks = n_tasks
        self.sample_tasks(n_tasks)
        self.max_episode_steps = max_episode_steps
        self.velocity = 0.
        super(HalfCheetahVelEnvMulti, self).__init__(env_args=env_args)
        self.logits = np.eye(n_tasks)
        self.task_logit = self.logits[0]
        self.reset_task(0)

    def revise_reward(self, info):
        # 根据step的info修正reward, 原来的reward越快越好, 现在的reward要保持速度
        # gym.envs.mujoco.half_cheetah.HalfCheetahEnv.step()返回的info
        # info = {"reward_run": (xposafter - xposbefore)/self.dt, 这个就是forward_vel
        #         "reward_ctrl": - 0.1 * np.square(action).sum()}

        # reward_old = reward_run + reward_ctrl
        # reward_new = reward_goal + reward_ctrl
        forward_vel = info["reward_run"]
        reward_task = -1.0 * abs(forward_vel - self._task["velocity"])
        # reward = 0.1 * info["reward_ctrl"] + reward_task  # 相应地缩减ctrl的权重
        reward = reward_task
        return reward

    def step(self, actions):
        reward, done, info = super().step(actions)
        self.velocity = info["reward_run"]
        reward = self.revise_reward(info)
        info["task"] = self._task["velocity"]
        info["is_goal_state"] = True if np.abs(info["reward_run"] - self._goal) <= 0.1 else False

        obs_n = self.get_obs()
        reward_n = [np.array(reward) for _ in range(self.n_agents)]
        return obs_n, reward_n, done, info

    def sample_tasks(self, num_tasks):
        velocities = np.linspace(start=1., stop=4., num=num_tasks)
        # velocities = np.array([3.0])
        self.tasks = [{"velocity": velocity} for velocity in velocities]

    # def set_task(self, task):
    #     self._task = task

    # def get_task(self):
    #     return self._task

    # def set_goal(self, goal):
    #     self._task["velocity"] = goal
    #     self._goal = goal
    #
    # def get_goal(self):
    #     return np.array(self._goal)

    def reset_task(self, idx=None):
        # 先进行任务重定向, 再环境重置
        if idx is None:
            idx = np.random.randint(self.num_tasks)
        self._task = self.tasks[idx]
        self._goal = self._task["velocity"]
        self.velocity = 0.
        self.task_logit = self.logits[idx]
        return self.reset()

    # def get_all_task_idx(self):
    #     return list(range(self.num_tasks))
    #
    # def set_all_goals(self, goals):
    #     assert self.num_tasks == len(goals)
    #     self.tasks = [{'velocity': velocity[0]} for velocity in goals]
    #     self.reset_task(0)


class HalfCheetahVelOracleEnvMulti(HalfCheetahVelEnvMulti):
    # def __init__(self, n_agents, n_tasks=10, max_episode_steps=1000, task={"velocity": 0.}, **kwargs):
    #     super(HalfCheetahVelOracleEnvMulti, self).__init__(n_agents, n_tasks, max_episode_steps, task, **kwargs)
    #     self.obs_size += 1
    #     self.observation_space = [Box(low=np.array([-10]*self.obs_size),
    #                                   high=np.array([10]*self.obs_size),
    #                                   dtype=np.float32) for _ in range(self.n_agents)]

    def step(self, actions):
        obs_n, reward_n, done, info = super().step(actions)
        task_obs = np.array([self.velocity - self._task["velocity"]])
        for i in range(self.n_agents):
            obs_n[i] = np.concatenate([self.task_logit, task_obs, obs_n[i]])
        return obs_n, reward_n, done, info

    def reset(self):
        obs_n = super().reset()
        task_obs = np.array([self.velocity - self._task["velocity"]])
        for i in range(self.n_agents):
            obs_n[i] = np.concatenate([self.task_logit, task_obs, obs_n[i]])
        return obs_n