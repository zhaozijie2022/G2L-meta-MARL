import importlib
import torchkit.pytorch_utils as ptu
from buffer.multi_task_replay_buffer import MultiTaskReplayBuffer
from buffer.separated_buffer import SeparatedReplayBuffer


def make_buffer(cfg):
    if "maddpg" in cfg.algo_file:
        rl_buffer = MultiTaskReplayBuffer(
            max_size=cfg.rl_buffer_size,
            obs_dim_n=cfg.obs_dim_n,
            action_dim_n=cfg.action_dim_n,
            max_episode_steps=cfg.max_ep_len,
            tasks=list(range(cfg.n_tasks)),
            n_agents=cfg.n_agents,
        )
    elif "mappo" in cfg.algo_file:
        rl_buffer = []
        for agent_id in range(cfg.n_agents):
            share_observation_space = (
                cfg.share_observation_space[agent_id]
                if cfg.use_centralized_V
                else cfg.observation_space[agent_id]
            )
            bu = SeparatedReplayBuffer(
                cfg,
                cfg.observation_space[agent_id],
                share_observation_space,
                cfg.action_space[agent_id],
            )
            rl_buffer.append(bu)
    else:
        raise NotImplementedError

    ae_buffer = MultiTaskReplayBuffer(
        max_size=cfg.ae_buffer_size,
        obs_dim_n=cfg.obs_dim_n,
        action_dim_n=cfg.action_dim_n,
        max_episode_steps=cfg.max_ep_len,
        tasks=list(range(cfg.n_tasks)),
        n_agents=cfg.n_agents,
    )
    return rl_buffer, ae_buffer




