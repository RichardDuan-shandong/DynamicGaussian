"""

    FileName          : gaussian_recon.py
    Author            : RichardDuan
    Last Update Date  : 2025-03-22
    Description       : backbone of the perception module
    Version           : 1.0
    License           : MIT License
    
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from perception_.depth_predictor.depth_predictor import SkeletonRecon 
from perception_.object.SkeletonGMM import SkeletonGMM
from voxel.voxel_grid import VoxelGrid
from perception_.gaussian_rendering.gaussian_regressor import GaussianRegressor
from helpers.network_utils import  MultiLayer3DEncoderShallow
from perception_.gaussian_rendering.gaussian_renderer import render
from perception_.utils import focal2fov, getProjectionMatrix, getWorld2View2
from typing import List

class SkeletonSplatter(nn.Module):
    def __init__(self, cfg, fabric):
        super().__init__()
        self.cfg = cfg
        self.fabric = fabric
        self.depth_predictor = SkeletonRecon(cfg=cfg, fabric=fabric)
    
    def forward(self, data):
        # 流程：1.先经过unet_embed得到feature_map
        #      2.在特征图上进行有偏的keypoint的采样，生成gaussian混合分布点(x, y, z, sigma, pi) 
        #      3.走一个DepthPredictor得到估计的深度，得到在世界坐标系中的guassian骨架
        #      4.根据骨架的GMM的三维分布进行采样，得到点云的先验分布    
        # 
        
        # (B, N, 2), (B, N, 2, 2), (B, N, 1), (B, N, 3), (B, N, 3, 3), (B, N, 1), (B, N), (B, 64, H, W)
        means_2d, cov_2d, weight_2d, means_3d, cov_3d, weight_3d, object_index, feature_map = self.depth_predictor(data)
            
        return means_3d, cov_3d, weight_3d, object_index, feature_map
    
class GaussianSampler():
    def __init__(self, cfg, fabric):
        self.cfg = cfg
        self.fabric = fabric
        
    def sample(self, data, means_3d, cov_3d, weight_3d, object_index): 
        pcds = []   # 点云先验分布 List : (N, 3)  --> length: batch_size*multiview_nums
        skeleton_gmm = [] # 经过mask筛选后的skeleton gmm分布 List
        pcds_project_to_image = []  # 点云投影回2d平面的坐标 :(N, 2) --> length:batch_size*multiview_nums
        for i in range(means_3d.shape[0]):
            means_3d_single_batch, cov_3d_single_batch, weight_3d_single_batch, object_index_single_batch = means_3d[i], cov_3d[i], weight_3d[i], object_index[i]
            GMM_3d = SkeletonGMM(
                means=means_3d_single_batch, 
                covariances=cov_3d_single_batch, 
                weights=weight_3d_single_batch, 
                object_index=object_index_single_batch, 
                fabric=self.fabric, 
                extr=data[i]['input_view']['pose'][0],
                intr=data[i]['input_view']['pose'][1],
                cfg=self.cfg)
            
            _pcds, _skeleton_gmm = GMM_3d.gmm_sample(self.cfg.skeleton_recon.recon_sample_num)    
            _pcds_project_to_image = GMM_3d.project_points(_pcds)
            
            pcds.append(_pcds)
            skeleton_gmm.append(_skeleton_gmm)
            pcds_project_to_image.append(_pcds_project_to_image)  # 左上角坐标系
            
        return pcds, pcds_project_to_image, skeleton_gmm

class GaussianRenderer(nn.Module):
    def __init__(self, cfg, fabric):
        super().__init__()
        self.cfg = cfg
        self.fabric = fabric
        
        self.bg_color = cfg.dataset.bg_color
        self._voxelizer = VoxelGrid(
        coord_bounds=cfg.rlbench.scene_bounds.cpu() if isinstance(cfg.rlbench.scene_bounds, torch.Tensor) else cfg.rlbench.scene_bounds,
        voxel_size=cfg.gaussian_renderer.voxel_sizes,
        device=fabric.device,  # 0
        batch_size=1,
        feature_size=0, # no rgb feat in 3d voxel
        max_num_coords=np.prod((cfg.rlbench.camera_resolution[0], cfg.rlbench.camera_resolution[1])),
    )
        if self.fabric is not None:
            self.voxel_encoder = self.fabric.setup_module(MultiLayer3DEncoderShallow(in_channels=7, out_channels=cfg.gaussian_renderer.final_dim))
        else:
            self.voxel_encoder = MultiLayer3DEncoderShallow(in_channels=7, out_channels=cfg.gaussian_renderer.final_dim)
        
        self.gaussian_regressor = GaussianRegressor(cfg=cfg, fabric=fabric)
        self.znear = cfg.cam.znear
        self.zfar = cfg.cam.zfar
        self.trans = cfg.dataset.trans # default: [0, 0, 0]
        self.scale = cfg.dataset.scale
        
    def get_rendering_calib(self, intr, extr):
        """
        get readable camera state for gaussian renderer from gt_pose
        :param data: dict
        :param data['intr']: intrinsic matrix (B, 3, 3)
        :param data['extr']: c2w matrix         (B, 4, 4)

        :return: dict
        """
        bs = intr.shape[0]
        device = intr.device
        fovx_list, fovy_list, world_view_transform_list, full_proj_transform_list, camera_center_list = [], [], [], [], []
        for i in range(bs):
            intr = intr[i, ...].cpu().numpy()
            extr = extr[i, ...].cpu().numpy()
            extr = np.linalg.inv(extr)  # the saved extrinsic is actually cam2world matrix, so turn it to world2cam matrix

            width, height = self.cfg.rlbench.camera_resolution[1], self.cfg.rlbench.camera_resolution[0]
            R = np.array(extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)    # inverse
            T = np.array(extr[:3, 3], np.float32)
            FovX = focal2fov(intr[0, 0], width)
            FovY = focal2fov(intr[1, 1], height)
            projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, K=intr, h=height, w=width).transpose(0, 1)
            world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(self.trans), self.scale)).transpose(0, 1) # [4, 4], w2c
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)    # [4, 4]
            camera_center = world_view_transform.inverse()[3, :3]   # inverse is c2w

            fovx_list.append(FovX)
            fovy_list.append(FovY)
            world_view_transform_list.append(world_view_transform.unsqueeze(0))
            full_proj_transform_list.append(full_proj_transform.unsqueeze(0))
            camera_center_list.append(camera_center.unsqueeze(0))

        novel_view_data = {
            'FovX': torch.FloatTensor(np.array(fovx_list)).to(device),
            'FovY': torch.FloatTensor(np.array(fovy_list)).to(device),
            'width': torch.tensor([width] * bs).to(device),
            'height': torch.tensor([height] * bs).to(device),
            'world_view_transform': torch.concat(world_view_transform_list).to(device),
            'full_proj_transform': torch.concat(full_proj_transform_list).to(device),
            'camera_center': torch.concat(camera_center_list).to(device),
        }

        return novel_view_data
    # 逐个batch循环
    def pts2render(self, x: dict, bg_color=[0,0,0]):
        '''
        x: 已经有x['xyz_maps'], ..., x['opacity_maps']
        '''
        '''use render function in GS'''
        # B=1
        xyz_0 = x['xyz_maps'][0, :, :]  # x['xyz_maps'] : [B, N, 3]
        feature_0 = x['sh_maps'][0, :, :, :]    # x['sh_maps'] : [B, N, 4, 3]
        rot_0 = x['rot_maps'][0, :, :]      # x['rot_maps'] : [B, N, 3, 3]
        scale_0 = x['scale_maps'][0, :, :]  # x['scale_maps'] : [B, N, 3, 1]
        opacity_0 = x['opacity_maps'][0, :, :]  # x['opcity_maps'] : [B, N, ?, ?]
        # feature_language_0 = data['feature_maps'][0, :, :]

        render_return_dict = render(
            x['rendering_calib'], 0, xyz_0, rot_0, scale_0, opacity_0, 
            bg_color=bg_color, pts_rgb=None, features_color=feature_0, features_language=None
            )

        x['img_pred'] = render_return_dict['render'].unsqueeze(0)    # 包含了一个batch所有的
        # data['novel_view']['embed_pred'] = render_return_dict['render_embed'].unsqueeze(0)
        return x
    
    def forward(self, data, pcds, pcds_project_to_image, feature_map):
        #      5. 点云生成voxel_grid，并编码获得voxel_feature             
        #      6. 点云对voxel_feature进行近邻采样（voxel_feature）
        #      7. 点云投影回二维平面，concat对应像素点上的feature_map的特征 (feature_map)
        #      8. 位置编码后concat(position_encoding)
        #      9. 经过一个encoder: (voxel_feature + image_feature + position_encoding) -> latent
        #      10. regressor复原gaussian
        #    注意：现在得到的点云是在特定相机拍摄角度下的，在训练过程中，如果要投影到其他相机视角上做监督，还是需要先将重建后的点云转换到world坐标系下，然后再进行渲染
        
        pcds_flat = [p.unsqueeze(0) for p in pcds]        
        voxel_grids = []
        
        # voxelize
        self._voxelizer = self._voxelizer.to(device=self.fabric.device)
        for _pcds_flat in pcds_flat:
            single_batch_voxel_grids, single_batch_voxel_density = self._voxelizer.coords_to_bounding_voxel_grid(
                _pcds_flat, coord_features=None, coord_bounds=torch.tensor(self.cfg.rlbench.scene_bounds).to(device=self.fabric.device), return_density=True)
            single_batch_voxel_grids = single_batch_voxel_grids.permute(0, 4, 1, 2, 3)
            voxel_grids.append(single_batch_voxel_grids)
            
        voxel_grids = torch.cat(voxel_grids, dim=0).detach().to(device=self.fabric.device)
        feature_map = feature_map.detach().to(device=self.fabric.device)
        
        # get 3d voxel feature
        voxel_feature, multi_scale_voxel_list = self.voxel_encoder(voxel_grids) # d0: [1, 128, 100, 100, 100] ([B,10,100,100,100] -> [B,128,100,100,100])
        
        # pcd regressor & render
        predict_render_images = []
        new_pcds = []
        gaussian_params_list = []
        
        debug_loss = 0
        for i in range(len(pcds)):  # 遍历各个batch
            # regressor
            _pcds = pcds_flat[i]
            _pcds_project_to_image = pcds_project_to_image[i].squeeze(0)
            _voxel_feature = voxel_feature[i]
            _image_feature = feature_map[i]
            gaussian_params, test_loss = self.gaussian_regressor(pcds=_pcds, 
                                                    pcds_project_to_image=_pcds_project_to_image.unsqueeze(0), 
                                                    voxel_feature=_voxel_feature.unsqueeze(0), 
                                                    image_feature=_image_feature.unsqueeze(0)
                                                    )   # dict
            
            debug_loss += test_loss
            
            new_pcds.append(gaussian_params['xyz_maps'][0])
            gaussian_params_list.append(gaussian_params)
            
            # render image
            _predict_render_image = None
            # multi_view_render (in every single batch)
            for j in range(len(data[i]['gt_view'])):    # 遍历一个batch内中各个监督视角
                
                extr_matrix = data[i]['gt_view'][j]['pose'][0].unsqueeze(0).to(device=gaussian_params['xyz_maps'].device)
                intr_matrix = data[i]['gt_view'][j]['pose'][1].unsqueeze(0).to(device=gaussian_params['xyz_maps'].device)
                
                gaussian_params['rendering_calib']=self.get_rendering_calib(extr=extr_matrix, intr=intr_matrix)
                gaussian_params = self.pts2render(x=gaussian_params, bg_color=self.bg_color)
                if _predict_render_image is not None:
                    _predict_render_image = torch.cat([_predict_render_image, gaussian_params['img_pred'].permute(0, 2, 3, 1)], dim=0)
                else:
                    _predict_render_image = gaussian_params['img_pred'].permute(0, 2, 3, 1)  
            
            _predict_render_image = _predict_render_image.unsqueeze(0)  # [1, multi_view_nums*(multi_view_nums-1), 128, 128, 3]
            predict_render_images.append(_predict_render_image)    # [1, N, 128, 128, 3]

        predict_render_images = torch.cat(predict_render_images, dim=0) # [B, N, 128, 128, 3]
        
        return predict_render_images, new_pcds, gaussian_params_list, debug_loss
    
class GaussianRecon(nn.Module):
    def __init__(self, cfg, fabric):
        super().__init__()
        self.cfg = cfg
        self.fabric = fabric
        
        self.bg_color = cfg.dataset.bg_color
        self._voxelizer = VoxelGrid(
        coord_bounds=cfg.rlbench.scene_bounds.cpu() if isinstance(cfg.rlbench.scene_bounds, torch.Tensor) else cfg.rlbench.scene_bounds,
        voxel_size=cfg.gaussian_renderer.voxel_sizes,
        device=fabric.device,  # 0
        batch_size=1,
        feature_size=0, # no rgb feat in 3d voxel
        max_num_coords=np.prod((cfg.rlbench.camera_resolution[0], cfg.rlbench.camera_resolution[1])),
    )
    
        self.voxel_encoder = MultiLayer3DEncoderShallow(in_channels=7, out_channels=cfg.gaussian_renderer.final_dim)
        
        self.gaussian_regressor = GaussianRegressor(cfg=cfg, fabric=fabric)
        self.depth_predictor = SkeletonRecon(cfg=cfg, fabric=fabric)
        self.znear = cfg.cam.znear
        self.zfar = cfg.cam.zfar
        self.trans = cfg.dataset.trans # default: [0, 0, 0]
        self.scale = cfg.dataset.scale
        
    def get_rendering_calib(self, intr, extr):
        """
        get readable camera state for gaussian renderer from gt_pose
        :param data: dict
        :param data['intr']: intrinsic matrix (B, 3, 3)
        :param data['extr']: c2w matrix         (B, 4, 4)

        :return: dict
        """
        bs = intr.shape[0]
        device = intr.device
        fovx_list, fovy_list, world_view_transform_list, full_proj_transform_list, camera_center_list = [], [], [], [], []
        for i in range(bs):
            intr = intr[i, ...].cpu().numpy()
            extr = extr[i, ...].cpu().numpy()
            extr = np.linalg.inv(extr)  # the saved extrinsic is actually cam2world matrix, so turn it to world2cam matrix

            width, height = self.cfg.rlbench.camera_resolution[1], self.cfg.rlbench.camera_resolution[0]
            R = np.array(extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)    # inverse
            T = np.array(extr[:3, 3], np.float32)
            FovX = focal2fov(intr[0, 0], width)
            FovY = focal2fov(intr[1, 1], height)
            projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, K=intr, h=height, w=width).transpose(0, 1)
            world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(self.trans), self.scale)).transpose(0, 1) # [4, 4], w2c
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)    # [4, 4]
            camera_center = world_view_transform.inverse()[3, :3]   # inverse is c2w

            fovx_list.append(FovX)
            fovy_list.append(FovY)
            world_view_transform_list.append(world_view_transform.unsqueeze(0))
            full_proj_transform_list.append(full_proj_transform.unsqueeze(0))
            camera_center_list.append(camera_center.unsqueeze(0))

        novel_view_data = {
            'FovX': torch.FloatTensor(np.array(fovx_list)).to(device),
            'FovY': torch.FloatTensor(np.array(fovy_list)).to(device),
            'width': torch.tensor([width] * bs).to(device),
            'height': torch.tensor([height] * bs).to(device),
            'world_view_transform': torch.concat(world_view_transform_list).to(device),
            'full_proj_transform': torch.concat(full_proj_transform_list).to(device),
            'camera_center': torch.concat(camera_center_list).to(device),
        }

        return novel_view_data
    # 逐个batch循环
    def pts2render(self, x: dict, bg_color=[0,0,0]):
        '''
        x: 已经有x['xyz_maps'], ..., x['opacity_maps']
        '''
        '''use render function in GS'''
        # B=1
        xyz_0 = x['xyz_maps'][0, :, :]  # x['xyz_maps'] : [B, N, 3]
        feature_0 = x['sh_maps'][0, :, :, :]    # x['sh_maps'] : [B, N, 4, 3]
        rot_0 = x['rot_maps'][0, :, :]      # x['rot_maps'] : [B, N, 3, 3]
        scale_0 = x['scale_maps'][0, :, :]  # x['scale_maps'] : [B, N, 3, 1]
        opacity_0 = x['opacity_maps'][0, :, :]  # x['opcity_maps'] : [B, N, ?, ?]
        # feature_language_0 = data['feature_maps'][0, :, :]

        render_return_dict = render(
            x['rendering_calib'], 0, xyz_0, rot_0, scale_0, opacity_0, 
            bg_color=bg_color, pts_rgb=None, features_color=feature_0, features_language=None
            )

        x['img_pred'] = render_return_dict['render'].unsqueeze(0)    # 包含了一个batch所有的
        # data['novel_view']['embed_pred'] = render_return_dict['render_embed'].unsqueeze(0)
        return x

    def forward(self, data):
        
        # 流程：1.先经过unet_embed得到feature_map
        #      2.在特征图上进行有偏的keypoint的采样，生成gaussian混合分布点(x, y, z, sigma, pi) 
        #      3.走一个DepthPredictor得到估计的深度，得到在世界坐标系中的guassian骨架
        #      4.根据骨架的GMM的三维分布进行采样，得到点云的先验分布
        
        
        means_2d, cov_2d, weight_2d, means_3d, cov_3d, weight_3d, object_index, feature_map = self.depth_predictor(data)
        
        pcds = []   # 点云先验分布 List : (N, 3)  --> length: batch_size*multiview_nums
        skeleton_gmm = [] # 经过mask筛选后的skeleton gmm分布 List
        pcds_project_to_image = []  # 点云投影回2d平面的坐标 :(N, 2) --> length:batch_size*multiview_nums
        for i in range(means_3d.shape[0]):
            means_3d_single_batch, cov_3d_single_batch, weight_3d_single_batch, object_index_single_batch = means_3d[i], cov_3d[i], weight_3d[i], object_index[i]
            GMM_3d = SkeletonGMM(
                means=means_3d_single_batch, 
                covariances=cov_3d_single_batch, 
                weights=weight_3d_single_batch, 
                object_index=object_index_single_batch, 
                fabric=self.fabric, 
                extr=data[i]['input_view']['pose'][0],
                intr=data[i]['input_view']['pose'][1],
                cfg=self.cfg)
            
            _pcds, _skeleton_gmm = GMM_3d.gmm_sample(self.cfg.skeleton_recon.recon_sample_num)    
            _pcds_project_to_image = GMM_3d.project_points(_pcds)
            
            pcds.append(_pcds)
            skeleton_gmm.append(_skeleton_gmm)
            pcds_project_to_image.append(_pcds_project_to_image)  # 左上角坐标系
            
        #      5. 点云生成voxel_grid，并编码获得voxel_feature             
        #      6. 点云对voxel_feature进行近邻采样（voxel_feature）
        #      7. 点云投影回二维平面，concat对应像素点上的feature_map的特征 (feature_map)
        #      8. 位置编码后concat(position_encoding)
        #      9. 经过一个encoder: (voxel_feature + image_feature + position_encoding) -> latent
        #      10. regressor复原gaussian
        #    注意：现在得到的点云是在特定相机拍摄角度下的，在训练过程中，如果要投影到其他相机视角上做监督，还是需要先将重建后的点云转换到world坐标系下，然后再进行渲染
        
        pcds_flat = [p.unsqueeze(0).to('cpu') for p in pcds]
        voxel_grids = []
        
        # voxelize
        for _pcds_flat in pcds_flat:
            single_batch_voxel_grids, single_batch_voxel_density = self._voxelizer.coords_to_bounding_voxel_grid(
                _pcds_flat, coord_features=None, coord_bounds=torch.tensor(self.cfg.rlbench.scene_bounds), return_density=True)
            single_batch_voxel_grids = single_batch_voxel_grids.permute(0, 4, 1, 2, 3).detach()
            
            voxel_grids.append(single_batch_voxel_grids)
        voxel_grids = torch.cat(voxel_grids, dim=0)
        
        # get 3d voxel feature
        voxel_feature, multi_scale_voxel_list = self.voxel_encoder(voxel_grids) # d0: [1, 128, 100, 100, 100] ([B,10,100,100,100] -> [B,128,100,100,100])
        
        # pcd regressor & render
        predict_render_images = []
        for i in range(len(pcds)):  # 遍历各个batch
            # regressor
            _pcds = pcds_flat[i]
            _pcds_project_to_image = pcds_project_to_image[i].squeeze(0)
            _voxel_feature = voxel_feature[i]
            _image_feature = feature_map[i]
            gaussian_params = self.gaussian_regressor(pcds=_pcds, 
                                                    pcds_project_to_image=_pcds_project_to_image.unsqueeze(0), 
                                                    voxel_feature=_voxel_feature.unsqueeze(0), 
                                                    image_feature=_image_feature.unsqueeze(0)
                                                    )   # dict
            
            _predict_render_image = None
            # multi_view_render (in every single batch)
            for j in range(len(data[i]['gt_view'])):    # 遍历一个batch内中各个监督视角
                
                extr_matrix = data[i]['gt_view'][j]['pose'][0].unsqueeze(0).to(device=gaussian_params['xyz_maps'].device)
                intr_matrix = data[i]['gt_view'][j]['pose'][1].unsqueeze(0).to(device=gaussian_params['xyz_maps'].device)
                
                gaussian_params['rendering_calib']=self.get_rendering_calib(extr=extr_matrix, intr=intr_matrix)
                gaussian_params = self.pts2render(x=gaussian_params, bg_color=self.bg_color)
                if _predict_render_image is not None:
                    _predict_render_image = torch.cat([_predict_render_image, gaussian_params['img_pred'].permute(0, 2, 3, 1)], dim=0)
                else:
                    _predict_render_image = gaussian_params['img_pred'].permute(0, 2, 3, 1)  
            
            _predict_render_image = _predict_render_image.unsqueeze(0)  # [1, multi_view_nums*(multi_view_nums-1), 128, 128, 3]
            predict_render_images.append(_predict_render_image)    # [1, N, 128, 128, 3]

        predict_render_images = torch.cat(predict_render_images, dim=0) # [B, N, 128, 128, 3]

        return predict_render_images, pcds, skeleton_gmm