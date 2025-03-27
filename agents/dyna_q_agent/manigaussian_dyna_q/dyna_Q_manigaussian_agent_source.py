import logging
import os
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from yarr.agents.agent import Agent, ActResult, ScalarSummary, \
    HistogramSummary, ImageSummary, Summary
from termcolor import colored, cprint
import io

from helpers.utils import visualise_voxel
from voxel.voxel_grid import VoxelGrid
from voxel.augmentation import apply_se3_augmentation_with_camera_pose
from helpers.clip.core.clip import build_model, load_clip
import PIL.Image as Image
import transformers
from helpers.optim.lamb import Lamb
from torch.nn.parallel import DistributedDataParallel as DDP
from agents.dyna_q_agent.manigaussian_dyna_q.neural_rendering import NeuralRenderer
from agents.dyna_q_agent.manigaussian_dyna_q.utils import visualize_pcd
from helpers.language_model import create_language_model

import wandb
import visdom
from lightning.fabric import Fabric

import numpy as np

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import MT30_V1

from torch.distributions import Normal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from torch.distributions import Categorical

class PPOActorCriticNetwork(nn.Module):
    def __init__(self, action_dim, hidden_dim=64):
        super(PPOActorCriticNetwork, self).__init__()

        # 使用卷积神经网络提取图像特征
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(6400, hidden_dim)  
        self.fc_mu = nn.Linear(hidden_dim, action_dim)  # 均值
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)  # 对数标准差

        # 价值网络 (critic)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # 卷积层
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # 展平
        x = x.reshape(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))

        # 输出连续动作的均值和对数标准差
        mu = self.fc_mu(x)
        log_std = self.fc_log_std(x)
        std = torch.exp(log_std)  # 标准差为对数标准差的指数

        # 价值网络的输出
        state_value = self.critic(x)

        return mu, std, state_value

class PPOAgent:
    def __init__(self, action_dim, lr=1e-3, gamma=0.99, epsilon=0.2, lambda_gae=0.95, clip_grad_norm=0.5, action_bounds=None):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_gae = lambda_gae
        self.action_bounds = action_bounds if action_bounds else [-0.25, -0.6, 0.8, 0.7, 0.65, 1.7]
        self.clip_grad_norm = clip_grad_norm

        # 初始化网络
        self.network = PPOActorCriticNetwork(action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

        # 存储经验
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.old_log_probs = []
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((84, 84)),  # 将图像调整为 84x84 大小
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
        ])
    def act(self, obs):
        """
        根据图像输入选择动作
        """
        # 处理图像数据
        image = self.transform(obs.front_rgb).unsqueeze(0)  # 处理图像并增加 batch 维度

        # 使用网络进行前向传播
        mu, std, value = self.network(image)

        # 创建高斯分布并从中采样
        dist = Normal(mu, std)
        action = dist.sample()  # 从高斯分布中采样
        action = self.format_action(action)
        print(action)
        # 返回采样的动作
        return action, value
    


    def format_action(self, action_probs):
        """
        格式整理动作，确保符合约束：
        1. XYZ 坐标被限制在给定的范围
        2. 四元数被归一化
        3. 夹爪状态被映射为 0 或 1
        """
        action = action_probs.squeeze().cpu().detach().numpy()

        # 1. XYZ 坐标约束，action的前三个数
        xyz = action[:3]  # 获取前三个数
        xyz_min = np.array(self.action_bounds[:3])
        xyz_max = np.array(self.action_bounds[3:6])
        xyz = np.clip(xyz, xyz_min, xyz_max)  # 限制在范围内

        # 2. 四元数归一化，action的中间四个数
        quat = action[3:7]  # 四元数的 4 个值
        quat = self.normalize_quaternion(quat)

        # 3. 夹爪状态，action的最后一个数
        gripper = 1 if action[7] > 0 else 0  # 使用阈值来判断夹爪开合

        # 合并新的动作
        action = np.concatenate([xyz, quat, [gripper]])

        return action

    def normalize_quaternion(self, q):
        """
        对四元数进行归一化处理
        """
        norm = np.linalg.norm(q)
        if norm < 1e-6:
            raise ValueError("Quaternion norm is too small to normalize.")
        return q / norm

    def store(self, state, action, reward, done, value):
        """
        存储经验
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def update(self):
        """
        PPO 的策略更新
        """
        advantages, returns = self.compute_advantages()
        
        states_tensor = torch.stack([torch.tensor(self.transform(state).unsqueeze(0), dtype=torch.float32) for state in self.states], dim=0)
        actions_tensor = torch.stack([torch.tensor(action, dtype=torch.float32) for action in self.actions], dim=0)


        actions = actions_tensor  # 对于连续动作，action是float
        old_log_probs = torch.tensor([log_prob.item() for log_prob in self.old_log_probs], dtype=torch.float32)
        
        for _ in range(10):  # PPO 中常使用多个 epoch
            mu, std, state_values = self.network(states_tensor)

            # 创建高斯分布
            dist = Normal(mu, std)

            # 计算当前动作的 log_prob
            log_probs = dist.log_prob(actions).sum(dim=-1)  # 对每个动作的 log_prob 求和

            # 计算熵
            entropy = dist.entropy().mean()

            # 计算 PPO 目标
            ratio = (log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages

            # 计算损失函数
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(state_values.squeeze(), returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            # 更新参数
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_grad_norm)
            self.optimizer.step()

        # 更新旧的 log_prob
        self.old_log_probs = log_probs

        # 清空存储的经验
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []

        
    def compute_advantages(self):
        """
        计算优势（advantages）和回报（returns）。
        """
        advantages = []
        returns = []
        gae = 0
        
        print(self.rewards)
        if len(self.rewards) < 2:
            return torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.float32)
        
        # 逆序计算 returns 和 advantages
        for step in reversed(range(len(self.rewards)-1)):
            # 计算TD残差
            delta = self.rewards[step] + self.gamma * self.values[step + 1] * (1 - self.dones[step]) - self.values[step]
            # 计算优势
            gae = delta + self.gamma * self.lambda_gae * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)

            # 计算回报
            return_ = advantages[0] + self.values[step]
            returns.insert(0, return_)

        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

# 使用示例
def operate():
    obs_config = ObservationConfig()
    obs_config.set_all(True)

    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(), gripper_action_mode=Discrete()),
        obs_config=ObservationConfig(),
        headless=False)
    env.launch()

    agent = PPOAgent(action_dim=8)

    train_tasks = MT30_V1['train']
    training_cycles_per_task = 3
    training_steps_per_task = 80
    episode_length = 40

    for _ in range(training_cycles_per_task):

        task_to_train = np.random.choice(train_tasks, 1)[0]
        task = env.get_task(task_to_train)
        task.sample_variation()  # random variation

        for i in range(training_steps_per_task):
            if i % episode_length == 0:
                print('Reset Episode')
                descriptions, obs = task.reset()
                print(descriptions)

            action, value = agent.act(obs)  # 根据图像选择动作

            obs, reward, done = task.step(action)

            agent.store(obs.front_rgb, action, reward, done, value)


            # 更新代理
            agent.update()


    print('Done')
    env.shutdown()
