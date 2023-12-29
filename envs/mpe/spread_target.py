import numpy as np
from gym.spaces import Box


class SpreadTargetMPE:
    def __init__(self, scenario_name, n_agents, n_tasks=2, max_episode_steps=100, task=None, success_factor=0.0, **kwargs):
        from envs.mpe.multiagent.environment import MultiAgentEnv
        from envs.mpe.multiagent import scenarios as scenarios
        scenario = scenarios.load(scenario_name + ".py").Scenario()
        world = scenario.make_world(n_agents, n_tasks)
        self.env = MultiAgentEnv(world=world,
                                 reset_callback=scenario.reset_world,
                                 reward_callback=scenario.reward,
                                 observation_callback=scenario.observation, )
        self.n_agents = n_agents  # 围捕者数量

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        share_obs_dim = sum([self.env.observation_space[i].shape[0] for i in range(self.n_agents)])
        self.share_observation_space = [Box(low=np.array([-np.inf] * share_obs_dim, dtype=np.float32),
                                            high=np.array([np.inf] * share_obs_dim, dtype=np.float32),
                                            dtype=np.float32) for _ in range(self.n_agents)]

        task = {"goal": [1, 1, 0]} if task is None else task
        self._task = task
        self._goal = task["goal"]
        self.num_tasks = n_tasks
        self.tasks = self.sample_tasks(n_tasks)
        self.max_episode_steps = max_episode_steps
        self.reset_task(0)

    def revise_reward(self, info):
        return

    def step(self, actions):
        # actions: List[np.ndarray] 只包含围捕者动作, 逃逸者动作为脚本生成
        obs_n, reward_n, done_n, info = self.env.step(list(actions))
        info["goal"] = self.get_goal()
        info["is_goal_state"] = True
        return obs_n, reward_n, done_n, info

    def sample_tasks(self, n_tasks):
        return [{"goal": [1, 1, 0]},
                {"goal": [1, 0, 1]},
                {"goal": [0, 1, 1]}]

    def reset_task(self, idx=None):
        if idx is None:
            idx = np.random.randint(self.num_tasks)
        self._task = self.tasks[idx]
        self._goal = self._task["goal"]
        for landmark in self.env.world.landmarks:
            l_id = int(landmark.name[-1])
            if self._goal[l_id] == 1:
                landmark.target = True
            else:
                landmark.target = False

    def reset(self):
        return self.env.reset()

    def render(self, mode="human"):
        return self.env.render(mode)

    def get_goal(self):
        return self._goal

