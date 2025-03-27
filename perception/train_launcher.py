"""

    FileName          : train_launcher.py
    Author            : RichardDuan
    Last Update Date  : 2025-03-08
    Description       : To initialize the train process(seed, fabric, cams etc.), and preprocess the data
    Version           : 1.0
    License           : MIT License
    
"""
import os
import numpy as np
import torch
import torch.distributed as dist
import lightning as L
import random
from perception.dataloader import _load_data
from perception.train import train

def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    
def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True  # 强制cuDNN使用确定性算法
    torch.backends.cudnn.benchmark = False  # 禁用 cuDNN 的优化

def run_seed(task_index, cfg, seed, device_list):

    # set conf
    set_random_seed(seed)
    cams = cfg.rlbench.cameras  # just use [front](maybe in the proceeding work we could try multiview-fusion method, but in GaussianSkeleton we just use a single front view)
    # rank = i
    # if fabric is not None:
    #     rank = fabric.global_rank
    # else:
    #     dist.init_process_group("gloo",
    #                     rank=rank,
    #                     world_size=cfg.ddp.num_devices)
        
    # build up save_dirs
    check_and_make(cfg.train.seg_save)  # data/data_temp
    
    data_origin = _load_data(cfg)
    
    train(data_origin, cfg, device_list)
    
    # 一个task的场景的训练完成
    task_index.value += 1            