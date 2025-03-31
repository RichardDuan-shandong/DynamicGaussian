import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
import torch.autograd.profiler as profiler

import os
import os.path as osp
import warnings
from termcolor import colored, cprint

from perception.gaussian_rendering.resnetfc import ResnetFC
from perception.gaussian_rendering.utils  import PositionalEncoding, visualize_pcd
from torch.nn.parallel import DistributedDataParallel as DDP

from typing import List
import numpy as np
import visdom

class GuassianDecoder(nn.Module):
    def __init__(self, cfg, out_channels, bias, scale):
        '''
        for weight initialization
        '''
        super().__init__()
        self.out_channels = out_channels
        self.cfg = cfg
        self.activation = torch.nn.functional.softplus
        self.out = nn.Linear(
            in_features=sum(out_channels),
            out_features=sum(out_channels),
        )
        start_channels = 0
        for out_channel, b, s in zip(out_channels, bias, scale):
            nn.init.xavier_uniform_(self.out.weight[start_channels:start_channels+out_channel, :], s)
            nn.init.constant_(
                self.out.bias[start_channels:start_channels+out_channel], b)
            start_channels += out_channel
            
    def forward(self, x):
        return self.out(self.activation(x, beta=100))
    


class GaussianEncoder(nn.Module):
    def __init__(self, cfg, out_channels):
        super().__init__()
        self.cfg = cfg
        
        self.coordinate_bounds = cfg.rlbench.scene_bounds # default: [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
        # print(colored(f"[GeneralizableNeRFEmbedNet] coordinate_bounds: {self.coordinate_bounds}", "red"))
        
        d_in = 3
        
        self.code = PositionalEncoding.from_conf(cfg["code"], d_in=d_in)
        
        d_in = self.code.d_out  # 39

        self.d_in = d_in

        self.image_shape = (cfg.rlbench.camera_resolution[0], cfg.rlbench.camera_resolution[1])
        self.num_objs = 0
        self.num_views_per_obj = 1
        # backbone
        self.d_latent = d_latent = [cfg.gaussian_renderer._3d_latent, cfg.gaussian_renderer._2d_latent] # 128
        self.d_lang = d_lang = cfg.gaussian_renderer.d_lang   # 128
        self.d_out = sum(out_channels)  
        

        self.encoder = ResnetFC(
                d_in=d_in, # xyz
                d_latent=sum(d_latent),  # representation dim
                d_lang=d_lang, 
                d_out=sum(out_channels), 
                d_hidden=cfg.gaussian_renderer.d_hidden, 
                n_blocks=cfg.gaussian_renderer.n_blocks, 
                combine_layer=cfg.gaussian_renderer.combine_layer,
                beta=cfg.gaussian_renderer.beta, use_spade=cfg.gaussian_renderer.use_spade,
            )

    def world_to_canonical(self, xyz):
        """
        :param xyz (B, N, 3) or (B, 3, N)
        :return (B, N, 3) or (B, 3, N)

        transform world coordinate to canonical coordinate with bounding box [0, 1]
        """
        xyz = xyz.clone()
        bb_min = self.coordinate_bounds[:3]
        bb_max = self.coordinate_bounds[3:]
        bb_min = torch.tensor(bb_min, device=xyz.device).unsqueeze(0).unsqueeze(0) if xyz.shape[-1] == 3 \
            else torch.tensor(bb_min, device=xyz.device).unsqueeze(-1).unsqueeze(0)
        bb_max = torch.tensor(bb_max, device=xyz.device).unsqueeze(0).unsqueeze(0) if xyz.shape[-1] == 3 \
            else torch.tensor(bb_max, device=xyz.device).unsqueeze(-1).unsqueeze(0)
        xyz -= bb_min
        xyz /= (bb_max - bb_min)

        return xyz

    def sample_in_canonical_voxel(self, xyz, voxel_feature):   # USED
        """
        :param xyz (B, 3)
        :param self.voxel_feat: [B, 128, 20, 20, 20]
        :return: (B, N, Feat)
        """
        xyz_voxel_space = xyz.clone()

        xyz_voxel_space = xyz_voxel_space * 2 - 1.0 # [0,1]->[-1,1]

        # unsqueeze the point cloud to also have 5 dim
        xyz_voxel_space = xyz_voxel_space.unsqueeze(1).unsqueeze(1) # [B, 1, 1, N, 3]
        
        # sample in voxel space
        point_3d_feature = F.grid_sample(voxel_feature, xyz_voxel_space, align_corners=True, mode='bilinear')    # [B, 128, 1, 1, N]

        # squeeze back to point cloud shape 
        point_3d_feature = point_3d_feature.squeeze(2).squeeze(2).permute(0, 2, 1)   # [bs, N, 128]

        return point_3d_feature # [B, N, 128]

    def sample_in_feature_map(self, project_xy, feature_map):
        """
        :param xyz: (B, N, 2)
        :param feature_map: (B, 64, H, W)
        :return: (B, N, Feat)
        """
        B, C, H, W = feature_map.shape
        xy_image_space = project_xy.clone()

        # 归一化到 [-1, 1]
        xy_image_space[:, :, 0] = xy_image_space[:, :, 0] / (W - 1) * 2 - 1  # x -> [-1, 1]
        xy_image_space[:, :, 1] = xy_image_space[:, :, 1] / (H - 1) * 2 - 1  # y -> [-1, 1]

        # 变成 grid_sample 需要的格式
        grid = xy_image_space.view(B, 1, -1, 2)  # (B, 1, N, 2)

        # 使用 grid_sample 进行插值
        point_2d_features = F.grid_sample(feature_map, grid, mode='bilinear', align_corners=True)
        
        # 调整形状
        point_2d_features = point_2d_features.squeeze(2).permute(0, 2, 1)  # (B, N, 64)

        return point_2d_features    # (B, N, Feat)
    
    def forward(self, pcds, pcds_project_to_image, voxel_feature, image_feature):
        """
        input: 
            pcds:(B, N, 3)
            pcds_project_to_image:(B, N, 2)
            voxel_feature:(B, 128, 20, 20, 20)
            image_feature:(B, 64, H, W)
        output:
        
        Predict gaussian parameter maps
        [Note] Typically B=1
        """
        B, N, _ = pcds.shape
        
        pcds = pcds.clone().detach()
        # canon_xyz = self.world_to_canonical(pcds)   # [1,N,3], min:-2.28, max:1.39
        # positional encoding
        position_xyz = pcds.reshape(-1, 3)  # (SB*B, 3)
        position_code = self.code(position_xyz)    # [N, 39]
        
        # volumetric sampling
        points_3d_feature = self.sample_in_canonical_voxel(pcds, voxel_feature) # [B, N, 128]
        points_3d_feature = points_3d_feature.reshape(-1, self.d_latent[0])  # [N, 128]

        # planar sampling

        points_2d_feature = self.sample_in_feature_map(pcds_project_to_image, image_feature)    # [B, N, 64]
        points_2d_feature = points_2d_feature.reshape(-1, self.d_latent[1])  # [N, 64]
        
        points_3d_feature = points_3d_feature
        points_2d_feature = points_2d_feature
        position_code     = position_code.to(device=points_3d_feature.device)
        
        points_input = torch.cat((points_3d_feature, points_2d_feature, position_code), dim=-1) # [N, 128+64+39]

        # Camera frustum culling stuff, currently disabled
        combine_index = None
        dim_size = None
        # backbone
        latent, _ = self.encoder(
            points_input,
            combine_inner_dims=(self.num_views_per_obj, N),
            combine_index=combine_index,
            dim_size=dim_size,
            language_embed=None,
            batch_size=B,
            )   # 26
        
        
        latent = latent.reshape(-1, N, self.d_out)  # [1, N, d_out]

        return latent
    
class GaussianRegressor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        split_dimensions, scale_inits, bias_inits = self._get_splits_and_inits(cfg)
        self.d_out = sum(split_dimensions)
        
        self.gs_encoder = GaussianEncoder(
            cfg,
            split_dimensions,
        )
        
        self.gs_decoder = GuassianDecoder(
            cfg,
            split_dimensions,
            scale=scale_inits,
            bias=bias_inits,
        )
        
        self.scaling_activation = torch.exp
        # self.scaling_activation = torch.nn.functional.softplus
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize    # [B, N, 4]
        self.max_sh_degree = cfg.gaussian_renderer.max_sh_degree


    def _get_splits_and_inits(self, cfg):
        '''Gets channel split dimensions and last layer initialization
        Credit: https://github.com/szymanowiczs/splatter-image/blob/main/scene/gaussian_predictor.py
        '''
        split_dimensions = []
        scale_inits = []
        bias_inits = []
        split_dimensions = split_dimensions + [3, 1, 3, 4, 3, 3]
        scale_inits = scale_inits + [
            cfg.gaussian_renderer.xyz_scale,
            cfg.gaussian_renderer.opacity_scale,
            cfg.gaussian_renderer.scale_scale,
            1.0,    # rotation
            1.0,    # feature_dc
            0.5,    # feature
            ]
        bias_inits = [
            cfg.gaussian_renderer.xyz_bias, 
            cfg.gaussian_renderer.opacity_bias,
            np.log(cfg.gaussian_renderer.scale_bias),
            0.0,
            0.1,
            0.05,
            ]
        if cfg.gaussian_renderer.max_sh_degree != 0:    # default: 1
            sh_num = (self.cfg.gaussian_renderer.max_sh_degree + 1) ** 2 - 1    # 3
            sh_num_rgb = sh_num * 3
            split_dimensions.append(sh_num_rgb)
            scale_inits.append(0.0)
            bias_inits.append(0.0)
        self.split_dimensions_with_offset = split_dimensions
        return split_dimensions, scale_inits, bias_inits


    def forward(self, pcds, pcds_project_to_image, voxel_feature, image_feature):
        """
        input: 
            pcds:(B, N, 3)
            pcds_project_to_image:(B, N, 2)
            voxel_feature:(B, 128, 20, 20, 20)
            image_feature:(B, 64, H, W)
        output:
        
        Predict gaussian parameter maps
        [Note] Typically B=1
        """

        # encode pcds
        latent = self.gs_encoder(pcds, pcds_project_to_image, voxel_feature, image_feature)
        
        # decode gaussian params
        split_network_outputs = self.gs_decoder(latent) # [1, N, (3, 1, 3, 4, 3, 9)]
        
        # convert output to gaussian params map
        split_network_outputs = split_network_outputs.split(self.split_dimensions_with_offset, dim=-1)
        
        xyz_maps, opacity_maps, scale_maps, rot_maps, features_dc_maps, feature_maps = split_network_outputs[:6]
        if self.max_sh_degree > 0:
            features_rest_maps = split_network_outputs[6]

        # spherical function head
        features_dc_maps = features_dc_maps.unsqueeze(2) #.transpose(2, 1).contiguous().unsqueeze(2) # [B, H*W, 1, 3]
        features_rest_maps = features_rest_maps.reshape(*features_rest_maps.shape[:2], -1, 3) # [B, H*W, 3, 3]
        sh_out = torch.cat([features_dc_maps, features_rest_maps], dim=2)  # [B, H*W, 4, 3]

        scale_maps = self.scaling_activation(scale_maps)    # exp
        scale_maps = torch.clamp_max(scale_maps, 0.05)
        
        # pcds = pcds.to(device=xyz_maps.device)
        
        # zero_tensor = torch.zeros_like(latent)
        # import torch.nn.functional as F
        # test_loss = F.l1_loss(latent, zero_tensor)
        gaussian_params = {}
        gaussian_params['xyz_maps'] = pcds + xyz_maps   # [B, N, 3]
        gaussian_params['sh_maps'] = sh_out    # [B, N, 4, 3]
        gaussian_params['rot_maps'] = self.rotation_activation(rot_maps, dim=-1)
        gaussian_params['scale_maps'] = scale_maps
        gaussian_params['opacity_maps'] = self.opacity_activation(opacity_maps)
        gaussian_params['feature_maps'] = feature_maps
        
        return gaussian_params