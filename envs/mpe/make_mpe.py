import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# 因为使用了imp, 在python3.4中就被废弃了, 隐藏warning


def make_mpe(scenario_name):
    """环境部分"""
    from envs.mpe.multiagent.environment import MultiAgentEnv
    from envs.mpe.multiagent import scenarios as scenarios
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world=world,
                        reset_callback=scenario.reset_world,
                        reward_callback=scenario.reward,
                        observation_callback=scenario.observation,)
    return env
















