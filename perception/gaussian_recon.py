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
from perception.depth_predictor.depth_predictor import SkeletonRecon 
from perception.object.SkeletonGMM import SkeletonGMM
from voxel.voxel_grid import VoxelGrid
from perception.gaussian_rendering.gaussian_regressor import GaussianRegressor
from helpers.network_utils import  MultiLayer3DEncoderShallow
from perception.gaussian_rendering.gaussian_renderer import render
from perception.utils import focal2fov, getProjectionMatrix, getWorld2View2
from typing import List
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
class SkeletonSplatter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.depth_predictor = SkeletonRecon(cfg)
    
    def forward(self, data):
        # 流程：1.先经过unet_embed得到feature_map
        #      2.在特征图上进行有偏的keypoint的采样，生成gaussian混合分布点(x, y, z, sigma, pi) 
        #      3.走一个DepthPredictor得到估计的深度，得到在世界坐标系中的guassian骨架
        #      4.根据骨架的GMM的三维分布进行采样，得到点云的先验分布    
        # 
        
        # (B, N, 2), (B, N, 2, 2), (B, N, 1), (B, N, 3), (B, N, 3, 3), (B, N, 1), (B, N), (B, 64, H, W)
        means_2d, cov_2d, weight_2d, means_3d, cov_3d, weight_3d, object_index, feature_map , image_color= self.depth_predictor(data)
            
        return means_3d, cov_3d, weight_3d, object_index, feature_map, image_color
    
def get_image_projected_points(data, pcds, cfg):
    pcds_project_to_image = []  # 点云投影回2d平面的坐标 :(N, 2) --> length:batch_size*multiview_nums
    for i in range(len(pcds)):
        _pcds = pcds[i]
        GMM_3d = SkeletonGMM(
                    means=None, 
                    covariances=None, 
                    weights=None, 
                    object_index=None, 
                    extr=data[i]['input_view']['pose'][0],
                    intr=data[i]['input_view']['pose'][1],
                    cfg=cfg)
        _pcds_project_to_image = GMM_3d.flip_axis(GMM_3d.project_points(_pcds))
        pcds_project_to_image.append(_pcds_project_to_image)    # 左上角坐标系
    return pcds_project_to_image
    
class GaussianSampler():
    def __init__(self, cfg):
        self.cfg = cfg
        
    def sample(self, data, means_3d, cov_3d, weight_3d, object_index): 
        pcds = []   # 点云先验分布 List : (N, 3)  --> length: batch_size*multiview_nums
        skeleton_gmm = [] # 经过mask筛选后的skeleton gmm分布 List
        for i in range(means_3d.shape[0]):
            means_3d_single_batch, cov_3d_single_batch, weight_3d_single_batch, object_index_single_batch = means_3d[i], cov_3d[i], weight_3d[i], object_index[i]
            GMM_3d = SkeletonGMM(
                means=means_3d_single_batch, 
                covariances=cov_3d_single_batch, 
                weights=weight_3d_single_batch, 
                object_index=object_index_single_batch, 
                extr=data[i]['input_view']['pose'][0],
                intr=data[i]['input_view']['pose'][1],
                cfg=self.cfg)
            
            _pcds, _skeleton_gmm = GMM_3d.gmm_sample(self.cfg.skeleton_recon.recon_sample_num)    
            
            pcds.append(_pcds)
            skeleton_gmm.append(_skeleton_gmm)
            
        return pcds, skeleton_gmm
def save_image(pcds_project_to_image, rgb_flat):
    import matplotlib.pyplot as plt
    import os
    save_dir = "output_images"   
    for i, (pcds, rgb) in enumerate(zip(pcds_project_to_image, rgb_flat)):
        pcds_ = np.array(pcds.cpu()).astype(int)  # (N, 2) 转 NumPy
        rgb_ = np.array(rgb.squeeze(0).cpu())  # (N, 3) 颜色信息

        image = np.zeros((128, 128, 3), dtype=np.float32)

        # 限制坐标范围，防止越界
        pcds_[:, 0] = np.clip(pcds_[:, 0], 0, 127)  # x 坐标
        pcds_[:, 1] = np.clip(pcds_[:, 1], 0, 127)  # y 坐标

        # 统计像素覆盖情况（防止多个点落在同一像素上）
        count = np.zeros((128, 128, 1), dtype=np.float32)

        # 绘制点云颜色
        for (x, y), color in zip(pcds_, rgb_):
            image[y, x] += color  # 注意：image[y, x]，因为 y 是行，x 是列
            count[y, x] += 1

        # 计算平均值，避免颜色覆盖问题
        count[count == 0] = 1
        image /= count
        
        # 显示并保存
        plt.imshow(image)
        plt.axis("off")  # 隐藏坐标轴
        plt.savefig(os.path.join(save_dir, f"image_{i}.png"), bbox_inches="tight", pad_inches=0)
        plt.close()  # 关闭当前图像，避免多次绘图叠加
        
class GaussianRegress(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.bg_color = cfg.dataset.bg_color
        self._voxelizer = VoxelGrid(
        coord_bounds=cfg.rlbench.scene_bounds.cpu() if isinstance(cfg.rlbench.scene_bounds, torch.Tensor) else cfg.rlbench.scene_bounds,
        voxel_size=cfg.gaussian_renderer.voxel_sizes,
        device=0,  # 0
        batch_size=1,
        feature_size=0, # no rgb feat in 3d voxel
        max_num_coords=np.prod((cfg.rlbench.camera_resolution[0], cfg.rlbench.camera_resolution[1])),
    )

        self.voxel_encoder = MultiLayer3DEncoderShallow(in_channels=10, out_channels=cfg.gaussian_renderer.final_dim)
        
        self.gaussian_regressor = GaussianRegressor(cfg)
    
    def forward(self, data, pcds, pcds_project_to_image, feature_map, image_color):
        #      5. 点云生成voxel_grid，并编码获得voxel_feature             
        #      6. 点云对voxel_feature进行近邻采样（voxel_feature）
        #      7. 点云投影回二维平面，concat对应像素点上的feature_map的特征 (feature_map)
        #      8. 位置编码后concat(position_encoding)
        #      9. 经过一个encoder: (voxel_feature + image_feature + position_encoding) -> latent
        #      10. regressor复原gaussian
        #    注意：现在得到的点云是在特定相机拍摄角度下的，在训练过程中，如果要投影到其他相机视角上做监督，还是需要先将重建后的点云转换到world坐标系下，然后再进行渲染
        
        pcds_flat = [p.unsqueeze(0) for p in pcds]        
        rgb_flat = []
        voxel_grids = []
        device = pcds_flat[0].device
        B, C, H, W = image_color.shape
        for j in range(B):
            single_batch_pcds_project_to_image = pcds_project_to_image[j]
            N, _ = single_batch_pcds_project_to_image.shape
            u = torch.clamp(single_batch_pcds_project_to_image[..., 0].long(), 0, W - 1)  # (B, N)
            v = torch.clamp(single_batch_pcds_project_to_image[..., 1].long(), 0, H - 1)  # (B, N)
            image_feature = image_color[j, :, v, u]  # (1, N, 3)
            rgb_flat.append(image_feature.permute(1, 0).unsqueeze(0))
            
        # save_image(pcds_project_to_image, rgb_flat)

        
        # voxelize
        for _pcds_flat, _rgb_flat in zip(pcds_flat, rgb_flat):
            single_batch_voxel_grids = self._voxelizer.coords_rgb_to_bounding_voxel_grid(
                _rgb_flat, _pcds_flat, coord_features=None, coord_bounds=torch.tensor(self.cfg.rlbench.scene_bounds).to(device=device))
            single_batch_voxel_grids = single_batch_voxel_grids.permute(0, 4, 1, 2, 3)
            voxel_grids.append(single_batch_voxel_grids)
            
        import pickle
        voxel_grids = torch.cat(voxel_grids, dim=0).detach().to(device=device)
        with open("voxel.pkl", "wb") as f:
           pickle.dump(voxel_grids[3].cpu(), f)
        #feature_map = feature_map.detach().to(device=device)
        
        # get 3d voxel feature
        voxel_feature, multi_scale_voxel_list = self.voxel_encoder(voxel_grids) # d0: [1, 128, 100, 100, 100] ([B,10,100,100,100] -> [B,128,100,100,100])
        
        # pcd regressor & render
        new_pcds = []
        gaussian_params_list = []
        
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
            
            new_pcds.append(gaussian_params['xyz_maps'][0])
            gaussian_params_list.append(gaussian_params)
            
        return new_pcds, gaussian_params_list
class GaussianRenderer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.bg_color = cfg.dataset.bg_color
        self.znear = cfg.cam.znear
        self.zfar = cfg.cam.zfar
        self.trans = cfg.dataset.trans # default: [0, 0, 0]
        self.scale = cfg.dataset.scale
        
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
        feature_language_0 = x['feature_maps'][0, :, :]
        
        render_return_dict = render(
            x['rendering_calib'], 0, xyz_0, rot_0, scale_0, opacity_0, 
            bg_color=bg_color, pts_rgb=None, features_color=feature_0, features_language=feature_language_0
            )

        x['img_pred'] = render_return_dict['render'].unsqueeze(0)    # 包含了一个batch所有的
        # data['novel_view']['embed_pred'] = render_return_dict['render_embed'].unsqueeze(0)
        return x
    
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
    
    def rendering(self, data, gaussian_params_list):
        # render image
        predict_render_images = []
        for i in range(len(data)):  # 遍历各个batch
            gaussian_params = gaussian_params_list[i]
            _predict_render_image = []
            # multi_view_render (in every single batch)
            for j in range(len(data[i]['gt_view'])):    # 遍历一个batch内中各个监督视角
                device = dist.get_rank()
                extr_matrix = data[i]['gt_view'][j]['pose'][0].to(device=device).unsqueeze(0)
                intr_matrix = data[i]['gt_view'][j]['pose'][1].to(device=device).unsqueeze(0)
                
                gaussian_params['rendering_calib']=self.get_rendering_calib(extr=extr_matrix, intr=intr_matrix)
                gaussian_params = self.pts2render(x=gaussian_params, bg_color=self.bg_color)

                _predict_render_image.append(gaussian_params['img_pred'].permute(0, 2, 3, 1))
                
            _predict_render_image = torch.cat(_predict_render_image, dim=0)
            _predict_render_image = _predict_render_image.unsqueeze(0)  # [1, multi_view_nums*(multi_view_nums-1), 128, 128, 3]
            predict_render_images.append(_predict_render_image)    # [1, N, 128, 128, 3]

        predict_render_images = torch.cat(predict_render_images, dim=0) # [B, N, 128, 128, 3]
        
        return predict_render_images