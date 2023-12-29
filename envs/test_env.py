# 测试环境能不能用的代码
from envs.mujoco.cheetah_dir import HalfCheetahDirEnvMulti
# from multi_goals.half_cheetah_vel import HalfCheetahVelEnvMulti
# from multi_goals.ant_goal import AntGoalEnvMulti
# from multi_goals.ant_angle import AntDirEnvMulti
#
# from multi_goals.half_cheetah_dir import HalfCheetahDirOracleEnvMulti
# from multi_goals.half_cheetah_vel import HalfCheetahVelOracleEnvMulti
# from multi_goals.ant_goal import AntGoalOracleEnvMulti
# from multi_goals.ant_angle import AntAngleOracleEnvMulti

# from rand_param_envs.hopper_rand_params import HopperRandParamsEnvMulti
# from rand_param_envs.walker2d_rand_params import Walker2dRandParamsEnvMulti
#
# from rand_param_envs.hopper_rand_params import HopperRandParamsOracleEnvMulti
# from rand_param_envs.walker2d_rand_params import Walker2dRandParamsOracleEnvMulti

import numpy as np
import time
import os

def main():
    n_agents = 6
    # env = make_env(env_type='multi_goals',
    #                env_name='half_cheetah_vel',
    #                env_class="HalfCheetahVelEnvMulti",
    #                n_agents=n_agents,
    #                n_tasks=3,
    #                max_episode_steps=100,
    #                seed=0, )
    env = HalfCheetahDirEnvMulti(n_agents=n_agents)
    env.set_task({"direction": -1.0})
    # env.set_task({"velocity": 0.0})

    # env_args = {"scenario": "HalfCheetah-v2",
    #             "agent_conf": "2x3",  # n_agents x motors_per_agent
    #             "agent_obsk": 0,
    #             "episode_limit": 1000}
    # env = MujocoMulti(env_args=env_args)

    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    n_episodes = 10

    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            obs_n = env.reset()
            # obs = env.get_obs()
            # state = env.get_state()

            actions = []
            for agent_id in range(n_agents):
                # avail_actions = env.get_avail_agent_actions(agent_id)
                # avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.uniform(-1.0, 1.0, n_actions)
                actions.append(action)

            pos_before = env.env.get_body_com("torso")[:2] * 1.
            obs_n, reward_n, terminated, info = env.step(actions)
            pos_after = env.env.get_body_com("torso")[:2]
            vel = (pos_after - pos_before) / env.env.dt
            print("vel_norm: %.4f, reward_ctrl: %.4f, reward_run: %.4f" %
                  (np.linalg.norm(vel), info["reward_ctrl"], info["reward_run"]))

            episode_reward += reward_n[0]

            time.sleep(0.1)
            print("reward_n = {}".format(reward_n))
            # env.render()


        print("Total reward in episode {} = {}".format(e, episode_reward))

    # env.close()

if __name__ == "__main__":
    # os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/home/zzj/.mujoco/mujoco210/bin'
    main()

