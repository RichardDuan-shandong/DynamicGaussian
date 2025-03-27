"""

    FileName          : depth_predictor.py
    Author            : RichardDuan
    Last Update Date  : 2025-03-16
    Description       : give the depth prediction on keypoint-gaussian pixel
    Version           : 1.0
    License           : MIT License
    
    *Notice           : SongUnet is not the contribution of GaussianDreamer.
                        But GaussianDreamer trys to make it to predict the depth on just several
                        keypoint pixel(namingly the position of 3d skeleton keypoint project on 2d image), 
                        thus make 3d representation more sparse.
    
"""
import torch
import torch.nn as nn
import torchvision

import numpy as np
import torch
from torch.nn.functional import silu
from torch.amp import autocast
from einops import rearrange, repeat
from perception.depth_predictor.unet_part_utils import Linear, Conv2d, PositionalEmbedding
from perception.depth_predictor.SongUnet import SongUNet
from perception.utils import fov2focal, focal2fov, world_to_canonical, canonical_to_world, getView2World
from sklearn.mixture import GaussianMixture
from perception.depth_predictor.keypoint_sample_utils import get_keypoint
from perception.object.SkeletonGMM import SkeletonGMM
# 使用Unet进行全图每个像素的Embedding
class UnetEmbedding(nn.Module):
    def __init__(self, cfg):
        super(UnetEmbedding, self).__init__()
        self.cfg = cfg
        
        in_channels = 3     # input:RGB
        
        self.encoder = SongUNet(cfg.rlbench.camera_resolution[0], 
                                in_channels, 
                                cfg.skeleton_recon.unet_embed_dim,
                                model_channels=cfg.skeleton_recon.base_dim,
                                num_blocks=cfg.skeleton_recon.num_blocks,
                                emb_dim_in=0,
                                channel_mult_noise=0,
                                attn_resolutions=cfg.skeleton_recon.attention_resolutions)
        
        # self.out = nn.Conv2d(in_channels=sum(out_channels), 
        #                      out_channels=sum(out_channels),
        #                      kernel_size=1)
        # start_channels = 0
        # for out_channel, b, s in zip(out_channels, bias, scale):
        #     nn.init.xavier_uniform_(
        #         self.out.weight[start_channels:start_channels+out_channel,
        #                         :, :, :], s)
        #     nn.init.constant_(
        #         self.out.bias[start_channels:start_channels+out_channel], b)
        #     start_channels += out_channel

    def forward(self, x, film_camera_emb=None, N_views_xa=1):
        x = self.encoder(x, 
                         film_camera_emb=film_camera_emb,
                         N_views_xa=N_views_xa)

        return x
    
class EmbeddingDecoder(nn.Module):
    def __init__(self, cfg, out_channels, scale ,bias):
        super().__init__()
        
        self.in_channels = cfg.skeleton_recon.unet_embed_dim
        self.out_channels = sum(out_channels)

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=32, kernel_size=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=self.out_channels, kernel_size=1)
        self.relu3 = nn.ReLU()
         
        self.regressor = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=1)
        
        # initialize param
        start_channels = 0
        for out_channel, b, s in zip(out_channels, bias, scale):
            nn.init.xavier_uniform_(
                self.regressor.weight[start_channels:start_channels+out_channel,
                                :, :, :], s)
            nn.init.constant_(
                self.regressor.bias[start_channels:start_channels+out_channel], b)
            start_channels += out_channel

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        
        x = self.regressor(x)
        return x
        

class SkeletonRecon(nn.Module):
    def __init__(self,
                 cfg,
                 device_ids):
        super().__init__()

        self.cfg = cfg
        self.device_ids = device_ids
        self.init_ray_dirs()
        split_dimensions, scale_inits, bias_inits = self.get_splits_and_inits(cfg)
        self.d_out = sum(split_dimensions)
        
        # self.unet_embed = UnetEmbedding(cfg, cfg.skeleton_recon.unet_embed_dim)

        self.encoder = torch.nn.DataParallel(UnetEmbedding(cfg), device_ids=device_ids)
        self.decoder = torch.nn.DataParallel(EmbeddingDecoder(cfg, split_dimensions, scale_inits, bias_inits), device_ids=device_ids)

    # 生成光线向量[dx, dy, 1] ->  [dx, dy, 1] / focal 从图片坐标系转换到相机坐标系
    '''
        ray_dirs[0, 0, :, :] → 代表 x 坐标
        ray_dirs[0, 1, :, :] → 代表 y 坐标
        ray_dirs[0, 2, :, :] → 全是 1，表示 z 方向
    '''
    def init_ray_dirs(self):
        x = torch.linspace(-self.cfg.rlbench.camera_resolution[0] // 2 + 0.5, 
                            self.cfg.rlbench.camera_resolution[0] // 2 - 0.5, 
                            self.cfg.rlbench.camera_resolution[0]) 
        y = torch.linspace( self.cfg.rlbench.camera_resolution[0] // 2 - 0.5, 
                        -self.cfg.rlbench.camera_resolution[0] // 2 + 0.5, 
                            self.cfg.rlbench.camera_resolution[0])

        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        ones = torch.ones_like(grid_x, dtype=grid_x.dtype)
        ray_dirs = torch.stack([grid_x, grid_y, ones]).unsqueeze(0)

        ray_dirs[:, :2, ...] /= abs(self.cfg.cam.focal)
        # on cuda:0
        self.register_buffer('ray_dirs', ray_dirs)  # (1, 3, H, W)

    def filter_ray(self, coords: torch.Tensor):
        """
        根据给定的像素坐标提取对应的光线方向。

        参数:
            coords (torch.Tensor): 形状 (B, N, 2) 的张量，每行是 (row, col)，表示图像中的像素坐标。

        返回:
            filtered_rays (torch.Tensor): 形状 (B, N, 3) 的张量，每个点对应的光线方向。
        """
        assert coords.shape[2] == 2, "坐标应当是 (B, N, 2) 形式的 (row, col) 对"

        # 获取 ray_dirs 的设备
        device = self.ray_dirs.device

        # 去掉 batch 维度，变成 (3, H, W)
        ray_dirs = self.ray_dirs.squeeze(0)  # 假设 ray_dirs 的形状是 (1, 3, H, W)
        
        # 获取 B, N
        B, N, _ = coords.shape
        
        # 提取 row 和 col，并确保它们在相同设备
        row = coords[:, :, 1].to(device).long()  # (B, N)
        col = coords[:, :, 0].to(device).long()  # (B, N)
        # 使用 advanced indexing 获取每个 batch 中的 ray_dirs 中对应的方向向量
        filtered_rays = ray_dirs[:, row, col].permute(1, 2, 0)  # (B, N, 3)

        return filtered_rays

    
    def get_splits_and_inits(self, cfg):
        # Gets channel split dimensions and last layer initialisation
        split_dimensions = []
        scale_inits = []
        bias_inits = []

        split_dimensions = split_dimensions + [1, 3, 2, 1]    # depth, offset_x/y/z, sigma_scale, pi_scale
        scale_inits = scale_inits + [
                        cfg.skeleton_recon.depth_scale, 
                        cfg.skeleton_recon.xyz_scale, 
                        cfg.skeleton_recon.cov_scale,
                        cfg.skeleton_recon.pi_scale
                        ]
        bias_inits = [cfg.skeleton_recon.depth_bias,
                        cfg.skeleton_recon.xyz_bias, 
                        cfg.skeleton_recon.cov_bias,
                        cfg.skeleton_recon.pi_bias]
        
        self.spit_dimensions_output = split_dimensions
        return split_dimensions, scale_inits, bias_inits

    def reshape_out(self, means, depth, offset, sigma_scale, pi_scale):
        x_indices = means[:, :, 0].to(device=means.device).long()  # (B, N)
        y_indices = means[:, :, 1].to(device=means.device).long()  # (B, N)

        B, C, H, W = depth.shape  # 例如 depth (B, 1, 128, 128)
        N = x_indices.shape[1]  # 采样点数

        def sample_from_map(tensor):
            """从 (B, C, H, W) 采样，返回 (B, N, C)"""
            B, C, H, W = tensor.shape  # 获取输入张量的形状
            
            # 展平 H, W 维度，变成 (B, C, H*W)
            tensor_flat = tensor.view(B, C, -1)  # (B, C, 128*128)

            # 计算扁平化索引
            index_flat = (y_indices * W + x_indices).view(B, 1, N).expand(-1, C, -1)  # (B, C, N)

            # 使用 gather 采样
            sampled = torch.gather(tensor_flat, 2, index_flat)  # (B, C, N)

            # 交换维度，变成 (B, N, C)
            return sampled.permute(0, 2, 1)  # (B, N, C)

        depth_sampled = sample_from_map(depth)  # (B, N, 1)
        offset_sampled = sample_from_map(offset)  # (B, N, 3)
        sigma_scale_sampled = sample_from_map(sigma_scale)  # (B, N, 2)
        pi_scale_sampled = sample_from_map(pi_scale)  # (B, N, 1)

        return depth_sampled, offset_sampled, sigma_scale_sampled, pi_scale_sampled

    
    
    # 流程：1.先经过unet_embed得到特征
    #      2.在特征图上进行有偏的keypoint的采样，生成gaussian混合分布点(x, y, z, sigma, pi) 
    #      3.走一个DepthPredictor得到估计的深度，得到在自己系中的guassian骨架
    #      4.在后面就是要采样，得到在这个相机视角下的点云先验分布
    def forward(self, x):
        # image: List -> Tensor:range[0, 1]
        images_list = [x[i]['input_view']['image'] for i in range(len(x))]
        images_list = [torch.tensor(img).permute(2, 0, 1).unsqueeze(0) if isinstance(img, (list, tuple)) or len(img.shape) == 3 else img for img in images_list]
        images_input = torch.cat(images_list, dim=0) # (B, 3, H, W)
        images_input = images_input.float() / 255.0

        # 1.sample keypoint with segemnt masks
        images_masks = []
        for i in range(len(x)):
            images_masks.append(x[i]['input_view']['masks'])
        # means:(B, N, 2) cov:(B, N, 2, 2) pi:(B, N, 1) object_index:(B, N) 
        # Note:N is obtained with padding, 'object_index' flags the padding part
        means, cov, pi, object_index = get_keypoint(images_masks, self.cfg, images_list=images_list)    # means在左上角坐标系（左手系）
        
        # 2.ray init
        # filtered_rays在相机坐标系(中心原点，右手系), 输入的means要变换为左下角坐标系（右手系）
        filtered_rays = self.filter_ray(means) # on cuda:0
        # 3.depth predict
        # embedding every pixel
        # （B, 64, H, W)
        # with ddp, image_input:cpu image_output:gpu
        feature_map = self.encoder(images_input)      # feature_map左上角坐标系
        
        # *cpu tensor to gpu tensor
        means, cov, pi, object_index = means.to(device=feature_map.device), cov.to(device=feature_map.device), pi.to(device=feature_map.device), object_index.to(device=feature_map.device)
        
        # spliter
        #  （B, 7, H, W） 7 = 1 + 3 + 2 + 1 (1是depth, 3是x/y/z_offset, 2是sigma_x/y_scale, 1是pi_scale)
        split_network_outputs = self.decoder(feature_map)
        split_network_outputs = split_network_outputs.split(self.spit_dimensions_output, dim = 1)   # 在 split_network_outputs上进行分割
         # depth:(B,1,128,128), offset:(B,3,128,128) sigma_scale:(B,2,128,128) pi_scale:(B,1,128,128)
        depth, offset, sigma_scale, pi_scale = split_network_outputs[:4] # 左上角坐标系
        # depth:(B,N,1), offset:(B,N,3) sigma_scale:(B,N,2) pi_scale:(B,N,1)  
        depth, offset, sigma_scale, pi_scale = self.reshape_out(means, depth, offset, sigma_scale, pi_scale) 
        
        sigma_scale = torch.sigmoid(sigma_scale) * 2  # (B, N, 2)
        pi_scale = torch.sigmoid(pi_scale) * 2  # (B, N, 1)
        
        # 4.spaltting-image in batch order
        B = depth.shape[0]
        means_3d = []
        cov_3d = []
        weight_3d = []
 
        for i in range(B):
            depth_single_view = depth[i]
            offset_single_view = offset[i]
            sigma_scale_single_view = sigma_scale[i]
            pi_scale_single_view = pi_scale[i]
            filtered_rays_single_view = filtered_rays[i]
            object_index_single_view = object_index[i]
            
            skeleton_gmm = SkeletonGMM(means=means[i], 
                                    covariances=cov[i], 
                                    weights=pi[i], 
                                    object_index=object_index[i], 
                                    extr=x[i]['input_view']['pose'][0],
                                    intr=x[i]['input_view']['pose'][1],
                                    cfg=self.cfg)
            
            means_3d_single_view, cov_3d_single_view, weight_3d_single_view = skeleton_gmm.image_spalt(
                                            filtered_rays=filtered_rays_single_view, 
                                            depth=depth_single_view, 
                                            offset=offset_single_view,
                                            sigma_scale=sigma_scale_single_view,
                                            pi_scale=pi_scale_single_view,
                                            object_index=object_index_single_view,
                                        )
            # _2d_project_single_view = skeleton_gmm.project_points(means_3d_single_view)
            
            # print(f'origin:{means[i]}')
            # print(f'projected:{_2d_project_single_view}')
            
            means_3d.append(means_3d_single_view.squeeze(0))
            cov_3d.append(cov_3d_single_view.squeeze(0))
            weight_3d.append(weight_3d_single_view.squeeze(0))
        
        if means_3d:
            means_3d = torch.stack(means_3d, dim=0)
            cov_3d = torch.stack(cov_3d, dim=0)
            weight_3d = torch.stack(weight_3d, dim=0)
        else:
            means_3d = torch.empty((0, 3))
            cov_3d = torch.empty((0, 3, 3))
            weight_3d = torch.empty((0, 1))
        # (B, N, 2), (B, N, 2, 2), (B, N, 1), (B, N, 3), (B, N, 3, 3), (B, N, 1), (B, N), (B, 64, H, W)
        return means, cov, pi, means_3d, cov_3d, weight_3d, object_index, feature_map