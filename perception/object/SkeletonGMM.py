import torch
import torch.nn as nn
import numpy as np
from perception_.utils import fov2focal, focal2fov, world_to_canonical, canonical_to_world, getView2World, getWorld2View2

# for every single view
class SkeletonGMM():
    # input: 2d
    def __init__(self, means:torch.Tensor, covariances:torch.Tensor, weights:torch.Tensor, object_index:torch.Tensor, extr, intr, cfg) ->None :
        self.means = means
        self.covariances = covariances
        self.weights = weights
        self.object_index = object_index
        self.extr = extr
        self.intr = intr
        self.depth_act = nn.Sigmoid()
        self.cfg = cfg
    """
        将 (N, 2, 2) 的协方差矩阵扩展为 (N, 3, 3) 的三维协方差矩阵。
        
        规则：
        1. 提取 covariances[:, 0, 0] 和 covariances[:, 1, 1]
        2. 乘以 sigma_scale[:, 0] 和 sigma_scale[:, 1] 进行缩放
        3. 非对角线的 [0,1] 和 [1,0] 乘以 sqrt(sigma_scale[:, 0] * sigma_scale[:, 1])
        4. 在 [2,2] 位置添加 1/2 * (covariances[:, 0, 0] + covariances[:, 1, 1])

        参数：
            covariances: (N, 2, 2) 的张量
            sigma_scale: (N, 2) 的缩放张量

        返回：pcd
            cov_3D: (N, 3, 3) 的三维协方差矩阵
    """
    def cov_expand_to_3D(self, sigma_scale):
        # 提取 sigma_x 和 sigma_y，并进行缩放
        sigma_x = torch.maximum(self.covariances[:, 0, 0] * sigma_scale[:, 0], torch.tensor(1e-6, device=self.covariances.device))
        sigma_y = torch.maximum(self.covariances[:, 1, 1] * sigma_scale[:, 1], torch.tensor(1e-6, device=self.covariances.device))

        # 计算非对角线项
        scale_xy = torch.sqrt(sigma_scale[:, 0] * sigma_scale[:, 1])
        cov_xy = torch.maximum(self.covariances[:, 0, 1] * scale_xy, torch.tensor(1e-6, device=self.covariances.device))  # 保证cov_xy合理

        # 计算新添加的 [2,2] 元素
        sigma_z = 0.5 * (sigma_x + sigma_y)

        # 初始化 (N, 3, 3) 的零矩阵
        cov_3D = torch.zeros((self.covariances.shape[0], 3, 3), dtype=self.covariances.dtype, device=self.covariances.device)

        # 赋值到新矩阵
        cov_3D[:, 0, 0] = sigma_x
        cov_3D[:, 1, 1] = sigma_y
        cov_3D[:, 0, 1] = cov_xy
        cov_3D[:, 1, 0] = cov_xy  # 对称性
        cov_3D[:, 2, 2] = sigma_z

        # 强制正定性，避免奇异矩阵
        epsilon = 1e-6  # 小正数
        cov_3D += torch.eye(3, device=cov_3D.device) * epsilon

        return cov_3D

    
    """
        计算单视角下的 3D 世界坐标点。

        参数：
            filtered_rays (torch.Tensor): 形状 (N, 3) 的光线方向向量 -> 经过means的筛选
            depth_network_single_view (torch.Tensor): 形状 (N, 1) 的深度值。
            offset (torch.Tensor): 形状 (N, 3) 的偏移量。
            extr (np.ndarray): 形状 (4, 4) 的相机外参矩阵。

        返回：
            pos (torch.Tensor): 形状 (N, 3) 的世界坐标点。
    """
    # filtered_rays:(N, 3) depth_network_single_view:(N, 1) offset:(N,3) extr: a single np of cam pose  
    def pos_expand_to_3D(self, filtered_rays, depth_network_single_view, offset):
        
        # 计算相机坐标系下的点坐标
        # 变为正
        depth = self.cfg.cam.znear + self.depth_act(depth_network_single_view) * (self.cfg.cam.zfar - self.cfg.cam.znear)   # (N, 1)
        # radius
        pos = filtered_rays * depth + offset # (N, 3)
        # 转换为齐次坐标 (N, 4)，第四维补 1
        ones = torch.ones((pos.shape[0], 1), dtype=pos.dtype, device=pos.device)  # (N, 1)
        pos_homogeneous = torch.cat([pos, ones], dim=-1)  # (N, 4)

        # c2w
        R = np.array(self.extr[:3, :3].to('cpu'), np.float32).T  # (3, 3) 变换矩阵
        T = np.array(self.extr[:3, 3].to('cpu'), np.float32)  # (3,) 平移向量
        c2w_matrix = getView2World(R, T)  # 获取 c2w 变换矩阵
        # 转换 `extr` 到 tensor
        c2w_matrix = torch.tensor(c2w_matrix, dtype=torch.float32, device=pos.device)  # (4, 4)
        # 变换到世界坐标 (N, 4)
        pos_world_homogeneous = pos_homogeneous @ c2w_matrix.T  # (N, 4)

        # 去掉最后一维 (w)，得到最终的 (N, 3)
        pos_world = pos_world_homogeneous[:, :3]  # (N, 3)

        pos_world = pos_world.unsqueeze(0)
        # 转换到 canonical 空间
        # pos_canonical = world_to_canonical(pos_world)  # (N, 3)

        return pos_world    # (N, 3)



    """
        扩展 weight 并根据 object_index 归一化。

        参数：
            weight (torch.Tensor): (N, 1) 代表权重        -                                  
            pi_scale (torch.Tensor): (N, 1) 代表放缩量
            object_index (torch.Tensor): (N,) 代表每个点所属的 object，-1 代表 padding

        返回：
            new_weight (torch.Tensor): (N, 1) 归一化后的权重
    """
    # weight (torch.Tensor): (N, 1) pi_scale (torch.Tensor): (N, 1) object_index (torch.Tensor): (N,)(代表每个点所属的 object，-1 代表 padding)
    def weight_expand_to_3D(self, pi_scale: torch.Tensor, object_index: torch.Tensor):

        # 放缩 weight
        scaled_weight = self.weights * pi_scale  # (N, 1)

        # 找到所有有效的 object（即 object_index != -1）
        mask_valid = object_index != -1

        # 创建一个和 weight 形状相同的全 0 张量
        new_weight = torch.zeros_like(scaled_weight)

        # 获取所有的唯一 object_id（不包括 -1）
        unique_objects = object_index[mask_valid].unique()

        # 遍历每个 object，进行归一化
        for obj_id in unique_objects:
            obj_mask = (object_index == obj_id)  # 当前 object 的 mask
            new_weight[obj_mask] = torch.softmax(scaled_weight[obj_mask], dim=0)  # 归一化

        return new_weight
    
    
    def image_spalt(self, filtered_rays, depth, offset, sigma_scale, pi_scale, object_index):
        
        self.means = self.pos_expand_to_3D(filtered_rays, depth, offset)
        self.covariances = self.cov_expand_to_3D(sigma_scale)
        self.weights = self.weight_expand_to_3D(pi_scale, object_index)
        
        return self.means, self.covariances, self.weights

    def flip_y_axis(self, points_2d):
        """
        对 2D 点的 y 坐标进行翻转，使其关于图像的水平中轴线对称。
        
        参数:
        - points_2d: (B, N, 2) 的 Tensor，存储 (x, y) 坐标
        - img_size: (H, W) 图像尺寸
        
        返回:
        - 转换后的 (B, N, 2) Tensor
        """
        H = self.cfg.rlbench.camera_resolution[0]-1 # 获取图像高度 H
        flipped_points = points_2d.clone()
        flipped_points[..., 1] = H - points_2d[..., 1]  # y' = H - y
        return flipped_points


    def project_points(self, _points_3d):
        
        """
        将 3D 点投影到 2D 图像平面（无 batch 维度）。

        参数:
        - points_3d: (N, 3) 的 Tensor，表示 N 个 3D 点
        - extr_np: (4, 4) 的 NumPy 数组，表示外参矩阵
        - intr_np: (3, 3) 的 NumPy 数组，表示内参矩阵
        - img_size: (H, W) 图像尺寸

        返回:
        - (N, 2) 的 Tensor，存储投影后的 2D 像素坐标
        """
        
        img_size=(self.cfg.rlbench.camera_resolution[0], self.cfg.rlbench.camera_resolution[0])
        
        points_3d = _points_3d.squeeze(0)
        N, _ = points_3d.shape

        # **1. NumPy 转 PyTorch**
        intr = torch.tensor(self.intr, dtype=torch.float32, device=points_3d.device)  # (3, 3)

        # **2. 扩展点为齐次坐标**
        ones = torch.ones((N, 1), device=points_3d.device)  # (N, 1)
        points_homo = torch.cat([points_3d, ones], dim=-1)  # (N, 4)

        # **3. 世界坐标 -> 相机坐标**
        R = np.array(self.extr[:3, :3], np.float32).T  # (3, 3) 变换矩阵
        T = np.array(self.extr[:3, 3], np.float32)  # (3,) 平移向量
        w2c = getWorld2View2(R, T)
        
        w2c= torch.tensor(w2c, dtype=torch.float32, device=points_homo.device)  
        points_camera_homo = points_homo @ w2c.T  # (N, 4)
        points_camera = points_camera_homo[..., :3]  # (N, 3)

        # **4. 透视投影**
        X_c, Y_c, Z_c = points_camera[:, 0], points_camera[:, 1], points_camera[:, 2]
        valid_mask = Z_c > 0  # 只投影在相机前面的点
        Z_c = torch.clamp(Z_c, min=1e-6)  # 避免除 0
        x_n = X_c / Z_c
        y_n = Y_c / Z_c

        # **5. 应用相机内参**
        fx, fy = abs(intr[0, 0]), abs(intr[1, 1])
        cx, cy = intr[0, 2], intr[1, 2]
        u = fx * x_n + cx
        v = fy * y_n + cy

        # **6. 限制范围**
        H, W = img_size
        u = torch.clamp(u, 0, W - 1)
        v = torch.clamp(v, 0, H - 1)

        # 标记无效点
        u[~valid_mask] = -1
        v[~valid_mask] = -1

        points_2d_projection = self.flip_y_axis(torch.stack([u, v], dim=-1))
        
        return points_2d_projection # (N, 2)
    
    # means_3d:(N, 3) cov_3d:(N, 3, 3) weight_3d：(N, 1) object_index:(N,)
    def gmm_sample(self, num_samples=500):
        """
        从多个 GMM（高斯混合分布）中进行采样，生成点云。
        
        Args:
            means_3d (Tensor): (N, 3) 每个高斯分布的均值
            cov_3d (Tensor): (N, 3, 3) 每个高斯分布的协方差矩阵
            weight_3d (Tensor): (N, 1) 每个高斯分布的权重，归一化
            object_index (Tensor): (N,) 物体索引，-1 代表 padding
            num_samples (int): 采样点数
        
        Returns:
            Tensor: (num_samples, 3) 采样得到的点云
        """
        valid_mask = self.object_index != -1  # 过滤掉 padding 部分
        means_mask = self.means[valid_mask]
        cov_mask = self.covariances[valid_mask]
        weight_mask = self.weights[valid_mask]
        object_index_mask = self.object_index[valid_mask]
        skeleton_gmm = {
            'means':means_mask,
            'cov':cov_mask,
            'weight':weight_mask
        }
        
        unique_objects = torch.unique(object_index_mask)  # 获取所有独立的 object_id
        sampled_points = []

        for obj_id in unique_objects:
            obj_mask = object_index_mask == obj_id
            obj_means = means_mask[obj_mask]
            obj_covs = cov_mask[obj_mask]
            obj_weights = weight_mask[obj_mask].squeeze(-1)  # (N_obj,)

            # 归一化权重，确保 sum=1
            obj_weights /= obj_weights.sum()

            # 根据权重进行类别采样
            num_obj_samples = int(num_samples * obj_weights.sum().item())  # 按比重分配样本数
            indices = torch.multinomial(obj_weights, num_obj_samples, replacement=True)  # 选高斯分布索引
            
            # 生成高斯分布采样点
            selected_means = obj_means[indices].float()
            selected_covs = obj_covs[indices].float()

            # 采样 (num_obj_samples, 3)
            sampled_points_obj = torch.distributions.MultivariateNormal(selected_means, selected_covs).sample()

            sampled_points.append(sampled_points_obj)

        # 合并所有 object 采样点
        pcd = torch.cat(sampled_points, dim=0)  # (num_samples, 3)

        return pcd, skeleton_gmm
    
    