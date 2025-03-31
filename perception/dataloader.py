"""

    FileName          : dataloader.py
    Author            : RichardDuan
    Last Update Date  : 2025-03-08
    Description       : Load data from local, here we use fabric to accelerate I/O loading
                        data.format:
    Version           : 1.0
    License           : MIT License
    
"""
import numpy as np
import PIL.Image as Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from lightning.fabric import Fabric
from tqdm import tqdm
from perception.const import MULTI_VIEW_DATA, SEG_FOLDER, EPISODE_FOLDER, EPISODES_FOLDER, VARIATIONS_ALL_FOLDER, POSE, IMAGES, DEPTHS, ZFAR, ZNEAR
from perception.utils import image_to_float_array

def parse_camera_file(file_path):
    """
    Parse our camera format.

    The format is (*.txt):
    
    4x4 matrix (camera extrinsic)
    space
    3x3 matrix (camera intrinsic)

    focal is extracted from the intrinsc matrix
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    camera_extrinsic = []
    for x in lines[0:4]:
        camera_extrinsic += [float(y) for y in x.split()]
    camera_extrinsic = np.array(camera_extrinsic).reshape(4, 4)

    camera_intrinsic = []
    for x in lines[5:8]:
        camera_intrinsic += [float(y) for y in x.split()]
    camera_intrinsic = np.array(camera_intrinsic).reshape(3, 3)

    focal = camera_intrinsic[0, 0]

    return camera_extrinsic, camera_intrinsic, focal

def parse_img_file(file_path, mask_gt_rgb=False, bg_color=[0,0,0,255]):
    """
    return np.array of RGB image with range [0, 1]
    """
    rgb = Image.open(file_path).convert('RGB')
    rgb = np.asarray(rgb).astype(np.float32)
    return rgb

def parse_depth_file(file_path):
    """
    return np.array of depth image
    """
    depth_img = Image.open(file_path).convert('RGB')
    depth = image_to_float_array(depth_img) * (ZFAR - ZNEAR) + ZNEAR   # meters
   # depth = np.asarray(depth).astype(np.float32)
    return depth

'''
    加载从仿真器中收集到的专家数据
'''
class CustomDataset(Dataset):
    def __init__(self, root_dir: str, num_episodes: int = None, cfg = None):
        """
        root_dir: 数据集的根目录
        num_episodes: 需要加载的 episode 数量，默认 None 加载所有
        """
        self.root_dir = root_dir
        self.cfg = cfg
        self.episodes = sorted(os.listdir(root_dir))  # 获取所有 episode
        if num_episodes:
            self.num_episodes = num_episodes  # 限制加载数量
        
        self.data_list = self._load_all_episodes()
        
    def _root_path_parser(self, root_dir, task):
        variation_path = os.path.join(
        root_dir, task,
        VARIATIONS_ALL_FOLDER)
        
        return variation_path
        
    def _load_all_episodes(self) -> List[Tuple[str, str]]:
        """解析所有 episode，收集 (image_path, pose_path) 对"""
        data_list = []
        tasks = self.cfg.rlbench.tasks
        for task in tasks:
            variation_path = self._root_path_parser(self.root_dir, task)
            for d in range(self.num_episodes):
                episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
                episode = EPISODE_FOLDER % d 
                episode_path = os.path.join(episodes_path, episode)
                multiview_path = os.path.join(episode_path, MULTI_VIEW_DATA)
                if not os.path.isdir(multiview_path):
                    continue  # 跳过非文件夹
                steps = sorted(os.listdir(multiview_path), key=int)  # 0,1,2,... 排序

                for step in steps:
                    step_path = os.path.join(multiview_path, step)
                    if not os.path.isdir(step_path):
                        continue  # 跳过非文件夹

                    images_folder = os.path.join(step_path, IMAGES)
                    poses_folder = os.path.join(step_path, POSE)
                    depths_folder = os.path.join(step_path, DEPTHS)
                    
                    if not os.path.exists(images_folder) or not os.path.exists(poses_folder):
                        print('Can\'t find local data!')
                        continue  # 如果缺少 images 或 poses，跳过
                    
                    image_files = sorted(os.listdir(images_folder))  # 获取所有视角的图片
                    pose_files = sorted(os.listdir(poses_folder))    # 获取所有视角的位姿
                    depth_files = sorted(os.listdir(depths_folder))    # 获取所有视角的深度图
                    
                    # 确保图片和位姿数量匹配
                    if len(image_files) != len(pose_files) or len(depth_files) != len(pose_files):
                        continue

                    for i in range(self.cfg.rlbench.num_view):
                        image_path = os.path.join(images_folder, f'{i}.png')
                        pose_path = os.path.join(poses_folder, f'{i}.txt')
                        depth_path = os.path.join(depths_folder, f'{i}.png')
                        # print(image_path)
                        data_list.append({
                            'image_path': image_path,
                            'depth_path': depth_path,
                            'pose_path': pose_path,
                            'task': task  # 保存路径对和对应的任务描述
                        })
                    #print('------------------------------------------------------')


        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_path, depth_path, pose_path, task_description = self.data_list[idx]['image_path'], self.data_list[idx]['depth_path'],self.data_list[idx]['pose_path'], self.data_list[idx]['task']
        image = parse_img_file(image_path)
        depth = parse_depth_file(depth_path)
        camera_extrinsic, camera_intrinsic, focal = parse_camera_file(pose_path)
        pose = [camera_extrinsic, camera_intrinsic, focal]
        return {'image': image, 'depth' : depth, 'pose': pose, 'description': task_description}


def _load_data(cfg):
    root_dir = cfg.rlbench.demo_path
    dataset = CustomDataset(root_dir, cfg.train.sample_trajectory_nums, cfg)
    # (dict('image' : 21, 'pose' : 21, 'description' : 21)))
    # to gurantee the original order of the data(every batch is means the multi_view at the same time)
    # shuffle = False num_workders = 0 is must!
    dataloader = DataLoader(
            dataset,
            batch_size=cfg.rlbench.num_view,    # every batch is means the multi_view at the same time
            shuffle=False,
            num_workers=3,      
            pin_memory=False
    )

    
    load_data_origin = {}
    single_task_data = []
    single_batch = []
    index = 0
    task_flag, last_task_flag = cfg.rlbench.tasks[0], cfg.rlbench.tasks[0]
    for multi_view_at_t in tqdm(dataloader, desc="Loading Data", unit="batch"):
        index = index + 1
        single_batch.append(multi_view_at_t)
        # 每cfg.train.batch_size步骤生成一个batch
        # ( (N, bs, dict('image' : 21, 'pose' : (21, 21, 21), 'description' : 21)) )
        if index % cfg.train.batch_size == 0 and index > 0:
            
            last_task_flag = task_flag
            task_flag = single_batch[0]['description'][0]
            if last_task_flag != task_flag or (task_flag == cfg.rlbench.tasks[-1] and (len(dataloader)-index)<cfg.train.batch_size): # 任务出现切换
                # print(last_task_flag) # debug : 显示添加的任务
                load_data_origin[last_task_flag] = single_task_data
                single_task_data = []
                
                
                break
                
                
                
            # 需要保证每个batch内都是同一个task下的一串序列的演示
            if single_batch[0]['description'][0] == single_batch[cfg.train.batch_size - 1]['description'][0]:
                single_task_data.append(single_batch)
                
                
            single_batch = []
        
    # 设计思考 ： 在train的过程中需要调控任务场景的训练，一次可以送入一个batch_size长度的步骤一起来重建训练，每一个步骤又有21个视角可用于训练和评估
    # --- 访问示例 ---
    # print(load_data_origin['close_jar'][0][0]['description'][0])    # 代表'close_jar'任务 第1组batch中第1个数据 任务描述(生成load_data_origin的时候已经保证每一组数据的的5个步骤都是同一个任务场景)
    # print(load_data_origin['close_jar'][0][0]['pose'])   # 代表'close_jar'任务 第1组batch中第1个数据   21个视角的pose 外参，内参，焦距(21, 21, 21)
    # print(load_data_origin['close_jar'][0][0]['image'])    # 代表'close_jar'任务 第1组batch中第1个数据   21个视角的图片 (21)

    return load_data_origin
