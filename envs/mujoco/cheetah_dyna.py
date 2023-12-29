"""adapted from https:github.com/cnfinn/maml_rl/rllab/envs/mujoco"""
import numpy as np
from envs.mujoco.multiagent_mujoco.mujoco_multi import MujocoMulti
# from . import RandParamsEnvMulti
# from gym import utils
# from . import RAND_PARAMS, RAND_PARAMS_EXTENDED


class HalfCheetahDynaEnvMulti(MujocoMulti):
    def __init__(self, n_agents=2, n_tasks=4, max_episode_steps=1000, task=None, **kwargs):
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

        task = {"dyna": 1.} if task is None else task
        self._task = task
        self._goal = task["dyna"]
        self.num_tasks = n_tasks
        self.tasks = self.sample_tasks(n_tasks)  # 需要super.init之后才能调用
        self.max_episode_steps = max_episode_steps
        super(HalfCheetahDynaEnvMulti, self).__init__(env_args=env_args)
        # self.default_params = {"body_mass": self.env.model.body_mass.copy()}  # 与阻止写入的数组不共享内存
        self.reset_task(0)

    def step(self, actions):
        reward, done, info = super().step(actions)
        info["goal"] = self.get_goal()
        info["is_goal_state"] = True
        obs_n = self.get_obs()
        reward_n = [reward * 1.0 for _ in range(self.n_agents)]
        done_n = [done for _ in range(self.n_agents)]
        return obs_n, reward_n, done_n, info

    def reset_task(self, idx=None):
        # ['body_mass', 'dof_damping', 'body_inertia', 'geom_friction']
        if idx is None:
            idx = np.random.randint(self.num_tasks)
        self._task = self.tasks[idx]
        bm = self.env.model.body_mass
        dd = self.env.model.dof_damping
        bi = self.env.model.body_inertia
        gf = self.env.model.geom_friction
        bm = bm / self._goal * self._task["dyna"]  # 除以还原mass, 再乘以新的mass
        dd = dd / self._goal * self._task["dyna"]
        bi = bi / self._goal * self._task["dyna"]
        gf = gf / self._goal * self._task["dyna"]
        self._goal = self._task["dyna"]
        self.reset()

    def sample_tasks(self, n_tasks):
        scales = np.linspace(start=0.5, stop=1.5, num=n_tasks)
        return [{"dyna": scale} for scale in scales]

    def set_task(self, task):
        self._task = task

    def get_task(self):
        return self._task

    def set_goal(self, goal):
        self._goal = goal
        self._task["dyna"] = goal

    def get_goal(self):
        return np.array(self._goal)

    def get_all_task_idx(self):
        return list(range(self.num_tasks))

