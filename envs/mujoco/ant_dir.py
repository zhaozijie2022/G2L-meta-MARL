import numpy as np

from envs.mujoco.multiagent_mujoco.mujoco_multi import MujocoMulti


class AntDirEnvMulti(MujocoMulti):
    def __init__(self, n_agents, n_tasks=3, max_episode_steps=100, task={"angle": 0.}, success_factor=0.0, **kwargs):
        if n_agents == 2:
            env_args = {"scenario": "Ant-v2",
                        "agent_conf": "2x4",
                        "agent_obsk": 1,
                        "episode_limit": max_episode_steps, }
        elif n_agents == 4:
            env_args = {"scenario": "Ant-v2",
                        "agent_conf": "4x2",
                        "agent_obsk": 1,
                        "episode_limit": max_episode_steps, }
        # elif n_agents == "2d":
        #     env_args = {"scenario": "Ant-v2",
        #                 "agent_conf": "2x4",
        #                 "agent_obsk": 1,
        #                 "episode_limit": max_episode_steps,}
        else:
            raise NotImplementedError("num of agents: %d doesn't match Ant" % n_agents)

        self._task = task
        self._goal = task["angle"]
        self.n_tasks = n_tasks
        self.sample_tasks(n_tasks)
        self.max_episode_steps = max_episode_steps
        self.velocity = np.zeros(2)
        self.success_factor = success_factor
        super(AntDirEnvMulti, self).__init__(env_args=env_args)
        self.logits = np.eye(n_tasks)
        self.task_logit = self.logits[0]
        self.reset_task(0)

    def revise_reward(self, info):
        # 1.鼓励更快速度, 2.惩罚更大的偏离角度
        # reward = info["angle_cos"] * info["vel_norm"] + info["reward_ctrl"] \
        #          + info["reward_contact"] + info["is_goal_state"] * self.success_factor
        reward = info["angle_cos"] * info["vel_norm"]
        # reward = reward_run * reward_angle
        return reward

    def step(self, actions):
        pos_before = self.env.get_body_com("torso")[:2] * 1.
        reward, done, info = super().step(actions)
        info["velocity"] = (self.env.get_body_com("torso")[:2] - pos_before) / self.env.dt
        info["vel_norm"] = np.linalg.norm(info["velocity"])
        info["angle_cos"] = np.dot(info["velocity"], self.angle_vector) / info["vel_norm"]
        info["is_goal_state"] = True if info["angle_cos"] > 0.965 else False
        info["task"] = self._task["angle"]

        reward = self.revise_reward(info)
        obs_n = self.get_obs()
        reward_n = [reward * 1.0 for _ in range(self.n_agents)]

        # return obs_n, reward_n, done, info
        return obs_n, reward_n, False, info

    def sample_tasks(self, num_tasks):
        angles = np.random.uniform(0., 2 * np.pi, size=(num_tasks,))
        self.tasks = [{"angle": angle} for angle in angles]

    # def set_task(self, task):
    #     self._task = task
    #     self._goal = task["angle"]
    #     self.angle_vector = np.array([1. * np.cos(self._goal), 1. * np.sin(self._goal)])
    #
    # def get_task(self):
    #     return self._task

    def reset_task(self, idx=None):
        # 先进行任务重定向, 再环境重置
        if idx is None:
            idx = np.random.randint(self.n_tasks)
        self._task = self.tasks[idx]
        self._goal = self._task["angle"]
        self.angle_vector = np.array([1. * np.cos(self._goal), 1. * np.sin(self._goal)])
        self.task_logit = self.logits[idx]
        return self.reset()

    # def get_all_task_idx(self):
    #     return list(range(self.n_tasks))
    #
    # def get_goal(self):
    #     return np.array(self._goal)


class AntAngleOracleEnvMulti(AntDirEnvMulti):
    def step(self, actions):
        obs_n, reward_n, done, info = super().step(actions)
        task_obs = np.concatenate(
            [info["velocity"] / np.linalg.norm(info["velocity"]) - self.angle_vector,
             np.array([np.linalg.norm(info["velocity"])])
             ])
        for i in range(self.n_agents):
            obs_n[i] = np.concatenate([self.task_logit, task_obs, obs_n[i]])
        return obs_n, reward_n, done, info

    def reset(self):
        obs_n = super().reset()
        task_obs = np.concatenate(
            [- self.angle_vector, np.zeros(1)]
        )
        for i in range(self.n_agents):
            obs_n[i] = np.concatenate([self.task_logit, task_obs, obs_n[i]])
        return obs_n













