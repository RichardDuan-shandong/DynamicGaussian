import os
import pickle
import gc
import logging
from typing import List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from rlbench import CameraConfig, ObservationConfig
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from train.model_based_train_runner import ModelBasedTrainRunner
from yarr.utils.stat_accumulator import SimpleAccumulator
from agents.dyna_q_agent import manigaussian_dyna_q
from helpers.custom_rlbench_env import CustomRLBenchEnv, CustomMultiTaskRLBenchEnv
import torch.distributed as dist

from agents.dyna_q_agent.manigaussian_dyna_q.envs_launch_utils.train_env_runner_launch import IndependentEnvRunner
from yarr.runners.env_runner import EnvRunner
from yarr.utils.rollout_generator import RolloutGenerator
from termcolor import cprint
import lightning as L
from tqdm import tqdm
from agents.dyna_q_agent.manigaussian_dyna_q.envs_launch_utils.custom_rlbench_env import CustomRLBenchEnv, CustomMultiTaskRLBenchEnv
from agents.dyna_q_agent.manigaussian_dyna_q.envs_launch_utils.train_offline_runner import OfflineTrainRunner
def run_seed(
        rank,
        cfg: DictConfig,
        obs_config: ObservationConfig,
        cams,
        multi_task,
        seed,
        world_size,
        env_config,
        fabric: L.Fabric = None,
) -> None:
    if fabric is not None:
        rank = fabric.global_rank
    else:
        dist.init_process_group("gloo",
                                rank=rank,
                                world_size=world_size)

    task = cfg.rlbench.tasks[0]
    tasks = cfg.rlbench.tasks  # 要执行哪些任务再yaml中定义

    rg = RolloutGenerator()
    replay_path = os.path.join(cfg.replay.path, 'seed%d' % seed)

    # 这里使用的dyna_Q由三阶段构成:

    if cfg.method.name == 'ManiGaussian_dyna_Q':
        # 创建一个缓冲区
        replay_buffer = manigaussian_dyna_q.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams, cfg.method.voxel_sizes,
            cfg.rlbench.camera_resolution,
            cfg=cfg)

        # 如果具有已经采集好的缓冲数据，则直接加载进缓冲区 缓冲区的路径为raplay_buffer : replay.path + seedxxx
        if cfg.replay.use_disk and (os.path.exists(replay_path) and len(os.listdir(replay_path)) > 1):  # default: True
            logging.info(f"Found replay files in {replay_path}. Loading...")
            replay_files = [os.path.join(replay_path, f) for f in os.listdir(replay_path) if f.endswith('.replay')]
            for replay_file in tqdm(replay_files,
                                    desc="Processing replay files"):  # NOTE: Experimental, please check your replay buffer carefully.
                with open(replay_file, 'rb') as f:
                    try:
                        replay_data = pickle.load(f)
                        replay_buffer._add(replay_data)
                    except pickle.UnpicklingError as e:
                        logging.error(f"Error unpickling file {replay_file}: {e}")
        else:

            manigaussian_dyna_q.launch_utils.fill_multi_task_replay(
                cfg, obs_config, 0,

                replay_buffer, tasks, cfg.rlbench.demos,  # RLBench 任务和演示数据(这里传入要执行哪些任务)

                cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,  # 是否使用数据增强
                cams, cfg.rlbench.scene_bounds,  # 相机参数和场景范围
                cfg.method.voxel_sizes, cfg.method.bounds_offset,  # 体素化（voxel）参数
                cfg.method.rotation_resolution, cfg.method.crop_augmentation,  # 旋转 & 裁剪增强
                keypoint_method=cfg.method.keypoint_method,  # 关键点检测方法
                fabric=fabric,  # 计算资源（Lightning Fabric）
            )

        agent = manigaussian_dyna_q.launch_utils.create_agent(cfg)




    else:
        raise ValueError('Method %s does not exists.' % cfg.method.name)

    wrapped_replay = PyTorchReplayBuffer(replay_buffer, num_workers=cfg.framework.num_workers)
    stat_accum = SimpleAccumulator(eval_video_fps=30)

    cwd = os.getcwd()
    weightsdir = os.path.join(cwd, 'seed%d' % seed, 'weights')  # load from the last checkpoint

    logdir = os.path.join(cwd, 'seed%d' % seed)

    cprint(f'Project path: {weightsdir}', 'cyan')

    # 第一阶段是使用专家数据进行模仿学习
    train_runner = OfflineTrainRunner(
        agent=agent,
        wrapped_replay_buffer=wrapped_replay,
        train_device=rank,
        stat_accumulator=stat_accum,
        iterations=cfg.framework.training_iterations,
        logdir=logdir,
        logging_level=cfg.framework.logging_level,
        log_freq=cfg.framework.log_freq,
        weightsdir=weightsdir,
        num_weights_to_keep=cfg.framework.num_weights_to_keep,
        save_freq=cfg.framework.save_freq,
        tensorboard_logging=cfg.framework.tensorboard_logging,
        csv_logging=cfg.framework.csv_logging,
        load_existing_weights=cfg.framework.load_existing_weights,
        rank=rank,
        world_size=world_size,
        cfg=cfg,
        fabric=fabric)
    cprint('Starting training!!', 'green')
    train_runner.start()
    # 第二阶段是将数据存入缓冲区

    # 定义训练环境
    train_env = CustomMultiTaskRLBenchEnv(
        task_classes=env_config[0],
        observation_config=env_config[1],
        action_mode=env_config[2],
        dataset_root=env_config[3],
        episode_length=env_config[4],
        headless=env_config[5],
        swap_task_every=env_config[6],
        include_lang_goal_in_obs=env_config[7],
        time_in_state=env_config[8],
        record_every_n=env_config[9])
    # 训练使用的环境交互器
    train_env_runner = IndependentEnvRunner(
        train_env=train_env,     # 要传入训练使用的环境
        agent=agent,
        train_replay_buffer=replay_buffer,
        num_train_envs=cfg.framework.train_envs,
        num_eval_envs=cfg.framework.eval_envs,                  # online evaluate
        rollout_episodes=99999,
        eval_episodes=cfg.framework.eval_episodes,
        training_iterations=cfg.framework.training_iterations,   # 训练的轮数
        eval_from_eps_number=cfg.framework.eval_from_eps_number, # 控制从哪个训练回合开始进行评估
        episode_length=cfg.rlbench.episode_length,
        stat_accumulator=stat_accum,
        weightsdir=weightsdir,
        logdir=logdir,
        env_device=cfg.framework.env_gpu,
        rollout_generator=rg,
        num_eval_runs=len(tasks),
        multi_task=multi_task)

    # dyna_q 算法是一种model-based的方法，使用model-based train runner
    train_runner = ModelBasedTrainRunner(
        agent=agent,
        env_runner=train_env_runner,
        wrapped_replay_buffer=wrapped_replay,
        train_device=rank,
        replay_buffer_sample_rates=None, # 使用均匀采样方案
        stat_accumulator=stat_accum,
        iterations=cfg.framework.training_iterations,
        num_train_envs = cfg.framework.train_envs,
        num_eval_envs = cfg.framework.eval_envs,
        eval_episode=cfg.framework.eval_episodes,
        logdir=logdir,
        log_freq=cfg.framework.log_freq,
        transitions_before_train=cfg.framework.transitions_before_train,
        weightsdir=weightsdir,
        num_weights_to_keep=cfg.framework.num_weights_to_keep,
        save_freq=cfg.framework.save_freq,
        replay_ratio=0.3,
        tensorboard_logging=cfg.framework.tensorboard_logging,
        csv_logging=cfg.framework.csv_logging,
        load_existing_weights=cfg.framework.load_existing_weights,
        rank=rank,
        world_size=world_size,
        cfg=cfg,
        fabric=fabric)

    cprint('Starting training!!', 'green')
    train_runner.start()

    del train_runner
    del agent
    gc.collect()
    torch.cuda.empty_cache()