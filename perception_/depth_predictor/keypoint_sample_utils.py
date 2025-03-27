"""

    FileName          : keypoint_sample_utils.py
    Author            : RichardDuan
    Last Update Date  : 2025-03-14
    Description       : to get sampled 2d keypoint according to sgement masks on images
    Version           : 1.0
    License           : MIT License
    
"""
from sklearn.mixture import GaussianMixture
import torch
import numpy as np

def pad_tensor(tensor_list, max_N, pad_value=0):
    """对 list 里的 tensor 进行 padding，使得 N 维度统一"""
    padded_list = []
    mask_list = []

    for tensor in tensor_list:
        if tensor is None:  # 无效数据，全部填充 pad_value
            padded_list.append(torch.full((max_N,) + tensor_list[0].shape[1:], pad_value))
            mask_list.append(torch.full((max_N,), -1, dtype=torch.int32))  # padding 部分填充 -1
        else:
            pad_size = max_N - tensor.shape[0]  # 计算需要填充的长度
            if pad_size > 0:
                padding = torch.full((pad_size,) + tensor.shape[1:], pad_value)
                padded_tensor = torch.cat([tensor, padding], dim=0)
                mask = torch.cat([torch.arange(1, tensor.shape[0] + 1), torch.full((pad_size,), -1)])  # 真实数据从 1 开始
            else:
                padded_tensor = tensor
                mask = torch.arange(1, tensor.shape[0] + 1)  # 全部有效

            padded_list.append(padded_tensor)
            mask_list.append(mask)

    return torch.stack(padded_list, dim=0), torch.stack(mask_list, dim=0)

# means:(B, N, 2) cov:(B, N, 2, 2) pi:(B, N, 1)
def get_keypoint_on_masks(images_masks, each_mask_sample_num=8, images_list=None):
    # 初始化 batch 级别的存储列表
    means_list = []
    covariances_list = []
    pi_list = []
    object_index_list = []
    max_N = 0  # 记录 batch 维度下的最大 N
    
    i = 0
    
    # 外层循环：遍历 batch 维度 (B)
    for single_image_masks in images_masks:
        
        # DEBUG
        # if images_list is not None:
        #     import cv2
        #     import os
        #     from PIL import Image
        #     image = images_list[i].clone()
            
        #     image = image.squeeze(0)
        #     image = image.permute(1, 2, 0).numpy()
        #     image = np.asarray(image).astype(np.uint8)
        #   #  print(image)
        #     i = i+1
        #     combined_mask = np.zeros((128,128), dtype=np.uint8)
        #     m = 0
        #     for single_object_masks in single_image_masks:
        #         if len(single_object_masks) <= 0:
        #             continue
        #         for image_mask in single_object_masks:
        #             m += 1
        #             combined_mask = np.logical_or(combined_mask, image_mask).astype(np.uint8)

        #     # 应用掩码
        #     masked_image = cv2.bitwise_and(image, image, mask=combined_mask)
        #     # 将 numpy 数组转换为 PIL 图像对象
        #     pil_image = Image.fromarray(masked_image)
        #     mask_path = os.path.abspath(f'mask_view_{i}——.png')
        #     # 保存图像
        #     pil_image.save(mask_path)
        
        #     print(f"Masked image saved to  {mask_path} {m}")
            
        batch_means = []
        batch_covariances = []
        batch_pi = []
        batch_object_index = []

        object_id = 1  # 每个 B 维度的 object 计数从 1 开始

        for single_object_masks in single_image_masks:
            if len(single_object_masks) <= 0:
                continue
            # 内层循环：遍历 batch 内的每个 object（mask）
            for image_mask in single_object_masks:
                # 获取所有值为 1 的坐标点
                points = np.argwhere((image_mask == 1) | (image_mask == True))
                points = points[:, [1, 0]] 
                # print(points)
                # 如果没有点符合条件（空 mask），跳过
                if len(points) == 0:
                    continue

                max_n = min(each_mask_sample_num, len(points) // 2)  # 不能超过样本点数的一半
                n_components_range = range(1, max_n + 1)  # 动态调整最大值

                bics = []

                # 尝试不同的组件数，找到最佳的 n_components
                for n in n_components_range:
                    gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
                    gmm.fit(points)
                    bics.append(gmm.bic(points))

                # 找到 BIC 最小的 n_components
                best_n = n_components_range[np.argmin(bics)]

                # 重新用最佳 n_components 训练 GMM
                gmm_best = GaussianMixture(n_components=best_n, covariance_type='full', random_state=42)
                gmm_best.fit(points)

                # 提取 GMM 参数
                means = torch.tensor(gmm_best.means_)  # (N, 2)
                covariances = torch.tensor(gmm_best.covariances_)  # (N, 2, 2)
                weights = torch.tensor(gmm_best.weights_).unsqueeze(-1)  # (N, 1)

                # 记录 object_id 信息
                object_index = torch.full((means.shape[0],), object_id, dtype=torch.int32)
                
                if len(batch_means) == 0:
                    batch_means = means
                    batch_covariances = covariances
                    batch_pi = weights
                    batch_object_index = object_index
                else:
                    batch_means = torch.cat([batch_means, means], dim=0)  # 拼接在N维度上
                    batch_covariances = torch.cat([batch_covariances, covariances], dim=0)
                    batch_pi = torch.cat([batch_pi, weights], dim=0)
                    batch_object_index = torch.cat([batch_object_index, object_index], dim=0)

                # 递增 object_id
                object_id += 1
            # 记录 batch 内的最大 N
            max_N = max(max_N, batch_means.shape[0])

        means_list.append(batch_means)
        covariances_list.append(batch_covariances)
        pi_list.append(batch_pi)
        object_index_list.append(batch_object_index)

    # 对所有 batch 进行 padding，使其 `N` 维度一致，并获取 padding mask
    padded_means, _ = pad_tensor(means_list, max_N, pad_value=0)
    padded_covariances, _ = pad_tensor(covariances_list, max_N, pad_value=0)
    padded_pi, _ = pad_tensor(pi_list, max_N, pad_value=0)
    padded_object_index, _ = pad_tensor(object_index_list, max_N, pad_value=-1)
    
    return padded_means, padded_covariances, padded_pi, padded_object_index

# means:(B, N, 2) cov:(B, N, 2, 2) pi:(B, N, 1)
def get_keypoint_on_non_masks(images_masks, n=12, epsilon_std=0.5):
    """
    在非mask区域上进行均匀网格采样，并加上扰动，构造高斯混合分布参数。

    参数：
        images_masks (list of list of np.ndarray): 形状 (B, N, H, W) 的二值 mask 列表。
        n (int): 采样点数，每个 batch 采样的点数。
        epsilon_std (float): 高斯扰动的标准差，用于模拟更自然的分布。

    返回：
        means (torch.Tensor): 形状 (B, n, 2)，扰动后的均匀网格采样坐标。
        covariances (torch.Tensor): 形状 (B, n, 2, 2)，单位协方差矩阵。
        pi (torch.Tensor): 形状 (B, n, 1)，混合权重（均匀分布）。
        object_index (torch.Tensor): 形状 (B, n)，物体索引（所有值为0，表示非mask区域）。
    """
    B = len(images_masks)
    means_list, covariances_list, pi_list, object_index_list = [], [], [], []
    H, W = images_masks[0][0][0].shape  # 获取图像尺寸

    for single_image_masks in images_masks:
        # 1. 计算 mask 之外的区域
        combined_mask = np.zeros((H, W), dtype=np.uint8)
        for single_object_masks in single_image_masks:
            if len(single_object_masks) <= 0:
                continue
            for image_mask in single_object_masks:  # 遍历 N 维度
                combined_mask = np.logical_or(combined_mask, image_mask)
        
        non_mask_coords = np.argwhere((combined_mask == 0))  # 获取未被 mask 覆盖的区域坐标
        non_mask_coords = non_mask_coords[:, [1, 0]]  # (x, y) 格式

        if len(non_mask_coords) == 0:
            sampled_points = np.zeros((n, 2))  # 如果没有可用点，则填充 0
        else:
            # 2. 生成均匀网格点
            num_grid_x = int(np.sqrt(n))
            num_grid_y = int(np.ceil(n / num_grid_x))

            x_linspace = np.linspace(0, W - 1, num_grid_x)
            y_linspace = np.linspace(0, H - 1, num_grid_y)
            grid_x, grid_y = np.meshgrid(x_linspace, y_linspace)
            grid_points = np.stack([grid_x.flatten(), grid_y.flatten()], axis=-1)

            # 3. 只保留非 mask 区域的点
            valid_points = []
            for px, py in grid_points:
                if combined_mask[int(py), int(px)] == 0:
                    valid_points.append([px, py])

            valid_points = np.array(valid_points)

            if len(valid_points) >= n:
                sampled_points = valid_points[:n]  # 直接选取前 n 个点
            else:
                padding = np.zeros((n - len(valid_points), 2))
                sampled_points = np.vstack([valid_points, padding])  # 补 0

        # 4. 在 sampled_points 上加上高斯扰动
        epsilon = np.random.normal(0, epsilon_std, sampled_points.shape)  # 生成高斯扰动
        disturbed_points = sampled_points + epsilon  # 施加扰动

        # 5. 生成高斯混合模型参数
        means = torch.tensor(disturbed_points, dtype=torch.float32)
        means = means.clamp(min=0, max=H-1)  # 确保在图像范围内
        covariances = torch.eye(2, dtype=torch.float32).repeat(n, 1, 1) # (n, 2, 2)
        pi = torch.full((n, 1), 1.0 / n, dtype=torch.float32)  # (n, 1)
        object_index = torch.zeros((n,), dtype=torch.int32)  # 物体索引，表示为 0

        means_list.append(means)
        covariances_list.append(covariances)
        pi_list.append(pi)
        object_index_list.append(object_index)

    # 6. 拼接 batch 维度
    means = torch.stack(means_list, dim=0)  # (B, N, 2)
    covariances = torch.stack(covariances_list, dim=0)  # (B, N, 2, 2)
    pi = torch.stack(pi_list, dim=0)  # (B, N, 1)
    object_index = torch.stack(object_index_list, dim=0)  # (B, N)
    
    return means, covariances, pi, object_index

def get_keypoint(images_masks, cfg, images_list=None):
    """
    综合 mask 区域和非 mask 区域的关键点采样。
    
    参数：
        images_masks (list of list of np.ndarray): 形状 (B, N, H, W) 的二值 mask 列表。
        n (int): 非 mask 区域采样的点数。
        
    返回：
        means (torch.Tensor): 形状 (B, N+non_N, 2)，合并后的关键点坐标。
        covariances (torch.Tensor): 形状 (B, N+non_N, 2, 2)，合并后的协方差矩阵。
        pi (torch.Tensor): 形状 (B, N+non_N, 1)，合并后的混合权重。
        object_index (torch.Tensor): 形状 (B, N+non_N)，合并后的 object 索引。
    """
    # 获取 mask 部分的关键点
    mask_means, mask_covariances, mask_pi, mask_object_index = get_keypoint_on_masks(images_masks, cfg.skeleton_recon.each_mask_sample_num, images_list)
    
    # 获取非 mask 部分的关键点
    non_mask_means, non_mask_covariances, non_mask_pi, non_mask_object_index = get_keypoint_on_non_masks(images_masks, cfg.skeleton_recon.non_masks_sample_num, cfg.skeleton_recon.non_masks_disturb_epsilon)
    
    # 合并 mask 和非 mask 部分
    means = torch.cat([non_mask_means, mask_means], dim=1)  # 合并在 N 维度上
    covariances = torch.cat([non_mask_covariances, mask_covariances], dim=1)  # 合并在 N 维度上
    pi = torch.cat([non_mask_pi, mask_pi], dim=1)  # 合并在 N 维度上
    object_index = torch.cat([non_mask_object_index, mask_object_index], dim=1)  # 合并在 N 维度上

    return means, covariances, pi, object_index