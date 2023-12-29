import os
import sys
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf

import torchkit.pytorch_utils as ptu
from learner import Learner


if __name__ == "__main__":

    env_cfg = OmegaConf.load("config/env_config/mpe-prey.yaml")
    algo_cfg = OmegaConf.load("config/algo_config/mappo.yaml")
    expt_cfg = OmegaConf.load("./config/expt.yaml")
    cfg = OmegaConf.merge(env_cfg, algo_cfg, expt_cfg)

    print("cuda is available: ", torch.cuda.is_available())
    ptu.set_gpu_mode(torch.cuda.is_available(), gpu_id=0)
    torch.set_num_threads(cfg.n_training_threads)

    cfg.use_linear_lr_decay = True
    cfg.save_model = False

    learner = Learner(cfg)
    learner.train()

