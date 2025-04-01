"""

    FileName          : train_gaussian_rendering.py
    Author            : RichardDuan
    Last Update Date  : 2025-03-08
    Description       : The entrance of gaussian_rendering pretraining, including basic config settings, it just loader one task once 
                        the config is managed by conf/perception/gaussian_rendering.yaml
    Version           : 1.0
    License           : MIT License
    
"""
import random
from perception.train_launcher import run_seed
from typing import List
from multiprocessing import Process, Manager
import os
import logging
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf, ListConfig
import torch
import lightning as L

from hydra.core.global_hydra import GlobalHydra
GlobalHydra.instance().clear()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


# 导入cfg参数配置,启动训练
@hydra.main(config_name='gaussian_rendering', config_path='conf')
def main(cfg: DictConfig) -> None:
    cfg_yaml = OmegaConf.to_yaml(cfg)
    # log the working dir
    cwd = os.getcwd()
    logging.info('CWD:' + cwd)
    os.environ['MASTER_ADDR'] = cfg.ddp.master_addr
    os.environ['MASTER_PORT'] = cfg.ddp.master_port
    # set seed specified
    if cfg.framework.start_seed >= 0:
        start_seed = cfg.framework.start_seed
    else:
        start_seed = 0

    seed_folder = os.path.join(os.getcwd(), 'seed%d' % start_seed)
    
    # different checkpoints is managed by its seed
    os.makedirs(seed_folder, exist_ok=True)
    with open(os.path.join(seed_folder, 'config.yaml'), 'w') as f:
        f.write(cfg_yaml)
        
    manager = Manager()
    result_dict = manager.dict()
    file_lock = manager.Lock()
    task_index = manager.Value('i', 0)
    variation_count = manager.Value('i', 0)
    lock = manager.Lock()
    
    world_size = cfg.ddp.num_devices

    #device_list = [i for i in range(world_size)]
    # use pytorch DDP
    #run_seed(task_index, cfg, start_seed, device_list)
    import torch.multiprocessing as mp
    mp.set_sharing_strategy('file_system')
    from torch.multiprocessing import set_start_method, get_start_method
    
    try:
        if get_start_method() != 'spawn':
            set_start_method('spawn', force=True)
    except RuntimeError:
        print("Could not set start method to spawn")
        pass
    mp.spawn(run_seed,
                args=(0,
                    cfg,
                    start_seed,
                    ),
                nprocs=world_size,
                join=True)
if __name__ == "__main__":
    main()