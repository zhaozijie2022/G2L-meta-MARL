import numpy as np
from gym.spaces import Box


class PreyTargetMPE:
    def __init__(self, scenario_name, n_agents, n_tasks=2, max_episode_steps=100, task=None, success_factor=0.0, **kwargs):
        from envs.mpe.multiagent.environment import MultiAgentEnv
        from envs.mpe.multiagent import scenarios as scenarios
        scenario = scenarios.load(scenario_name + ".py").Scenario()
        world = scenario.make_world(n_agents, n_tasks, n_obstacles=0)
        self.env = MultiAgentEnv(world=world,
                                 reset_callback=scenario.reset_world,
                                 reward_callback=scenario.reward,
                                 observation_callback=scenario.observation, )
        self.n_agents = n_agents  # 围捕者数量

        self.action_space = self.env.action_space[:self.n_agents]
        self.observation_space = self.env.observation_space[:self.n_agents]
        share_obs_dim = sum([self.env.observation_space[i].shape[0] for i in range(self.n_agents)])
        self.share_observation_space = [Box(low=np.array([-np.inf] * share_obs_dim, dtype=np.float32),
                                            high=np.array([np.inf] * share_obs_dim, dtype=np.float32),
                                            dtype=np.float32) for _ in range(self.n_agents)]

        task = {"goal": 0} if task is None else task
        self._task = task
        self._goal = task["goal"]
        self.num_tasks = n_tasks
        self.tasks = self.sample_tasks(n_tasks)
        self.max_episode_steps = max_episode_steps
        self.success_factor = success_factor
        self.reset_task(0)

    def revise_reward(self, info):
        return

    def step(self, actions):
        # actions: List[np.ndarray] 只包含围捕者动作, 逃逸者动作为脚本生成
        action_preys = []
        for agent in self.env.agents:
            if "prey" in agent.name:
                action_preys.append(self.prey_action(agent))
        action_env = list(actions) + action_preys
        obs_n, reward_n, done_n, info = self.env.step(action_env)
        info["goal"] = self.get_goal()
        info["is_goal_state"] = True
        return obs_n[:self.n_agents], reward_n[:self.n_agents], done_n[:self.n_agents], info

    def sample_tasks(self, n_tasks):
        return [{"goal": i} for i in range(n_tasks)]

    def reset_task(self, idx=None):
        if idx is None:
            idx = np.random.randint(self.num_tasks)
        self._task = self.tasks[idx]
        self._goal = self._task["goal"]
        for agent in self.env.agents:
            agent.target = True if agent.name == 'prey %d' % self._goal else False

    def reset(self):
        return self.env.reset()[:self.n_agents]

    def render(self, mode="human"):
        return self.env.render(mode)

    def get_goal(self):
        return self._goal

    def prey_action(self, prey):
        # TODO 逃逸者动作生成脚本, 连续动作空间
        action = np.random.rand(self.env.world.dim_p)
        if not prey.target:
            return action

        min_dist = np.inf
        for agent in self.env.agents:
            if "predator" in agent.name:
                delta_pos = prey.state.p_pos - agent.state.p_pos
                dist = np.linalg.norm(delta_pos)
                if dist < min_dist:
                    min_dist = dist
                    action = delta_pos / dist
        return action

    def close(self):
        self.env.close()

