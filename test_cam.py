import torch.nn as nn
import numpy as np
import torch

image_resolution = 128
depth_act = nn.Sigmoid()


def init_ray_dirs():
    x = torch.linspace(-image_resolution // 2 + 0.5, 
                        image_resolution // 2 - 0.5, 
                        image_resolution) 
    y = torch.linspace( image_resolution // 2 - 0.5, 
                    -image_resolution // 2 + 0.5, 
                        image_resolution)

    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    ones = torch.ones_like(grid_x, dtype=grid_x.dtype)
    ray_dirs = torch.stack([grid_x, grid_y, ones]).unsqueeze(0)

    ray_dirs[:, :2, ...] /= abs(110.851248)

    return ray_dirs

def getView2World(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = C2W
    return np.float32(Rt)

def world_to_canonical(xyz, coordinate_bounds=[-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]):
    """
    :param xyz (B, N, 3) or (B, 3, N)
    :return (B, N, 3) or (B, 3, N)

    transform world coordinate to canonical coordinate with bounding box
    """
    xyz = xyz.clone()
    bb_min = coordinate_bounds[:3]
    bb_max = coordinate_bounds[3:]
    bb_min = torch.tensor(bb_min, device=xyz.device).unsqueeze(0).unsqueeze(0) if xyz.shape[-1] == 3 \
        else torch.tensor(bb_min, device=xyz.device).unsqueeze(-1).unsqueeze(0)
    bb_max = torch.tensor(bb_max, device=xyz.device).unsqueeze(0).unsqueeze(0) if xyz.shape[-1] == 3 \
        else torch.tensor(bb_max, device=xyz.device).unsqueeze(-1).unsqueeze(0)
    xyz -= bb_min
    xyz /= (bb_max - bb_min)

    return xyz

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)
'''

rays position:

       (-127.5, 127.5)       (0, 127.5)       (127.5, 127.5)
          +---------------------+--------------------+
          |                     |                    |
          |                     |                    |
 (-127.5, 0) +----------------- (0,0) --------------+ (127.5, 0)
          |                     |                    |
          |                     |                    |
          +---------------------+--------------------+
       (-127.5, -127.5)      (0, -127.5)      (127.5, -127.5)

'''
znear = 2
zfar = 4.0

def pos_expand_to_3D(filtered_rays, depth_network_single_view, extr):
    
    # 计算相机坐标系下的点坐标
    # 变为正
    depth = znear + depth_act(depth_network_single_view) * (zfar - znear)   # (N, 1)

    # radius
    pos = filtered_rays * depth  # (N, 3)
    # 转换为齐次坐标 (N, 4)，第四维补 1
    ones = torch.ones((pos.shape[0], 1), dtype=pos.dtype, device=pos.device)  # (N, 1)
    pos_homogeneous = torch.cat([pos, ones], dim=-1)  # (N, 4)

    # c2w
    R = np.array(extr[:3, :3], np.float32).T  # (3, 3) 变换矩阵
    T = np.array(extr[:3, 3], np.float32)  # (3,) 平移向量
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

def filter_ray_(ray_dirs, coords: torch.Tensor):
        assert coords.shape[2] == 2, "坐标应当是 (B, N, 2) 形式的 (row, col) 对"

        # 去掉 batch 维度，变成 (3, H, W)
        ray_dirs = ray_dirs.squeeze(0)  # 假设 ray_dirs 的形状是 (1, 3, H, W)
        
        # 获取 B, N
        B, N, _ = coords.shape
        
        # 提取 row 和 col，并确保它们在相同设备
        row = coords[:, :, 1].long()  # (B, N)
        col = coords[:, :, 0].long()  # (B, N)
        # 使用 advanced indexing 获取每个 batch 中的 ray_dirs 中对应的方向向量
        filtered_rays = ray_dirs[:, row, col].permute(1, 2, 0)  # (B, N, 3)

        return filtered_rays
    
    
def generate_depth(N):
    """
    生成形状为 (N, 1) 的张量，范围在 (-5, 5) 之间。

    参数:
    - N: 点的数量

    返回:
    - depth: (N, 1) 的张量，范围在 (-5, 5) 之间
    """
    depth = torch.rand((N, 1)) * 10 - 5  # 生成 [0,1) 的随机数，然后映射到 (-5, 5)
    return depth

def single_batch_2d_to_3d(filter_ray, depth, extr):

    return pos_expand_to_3D(filter_ray, depth, extr)
    
    
def generate_2d_points(B, N, coordinate_bounds):
    """
    生成 (B, N, 2) 形状的 2D 坐标点，范围受 coordinate_bounds 限制。

    参数:
    - B: 批量大小
    - N: 每个 batch 生成的点数
    - coordinate_bounds: [x_min, y_min, x_max, y_max]

    返回:
    - points_2d: (B, N, 2) 2D 坐标
    """
    x_min, y_min, x_max, y_max = coordinate_bounds
    points_2d = torch.rand((B, N, 2), dtype=torch.float32)  # 先生成 [0,1] 之间的随机数
    points_2d[..., 0] = points_2d[..., 0] * (x_max - x_min) + x_min  # x 范围
    points_2d[..., 1] = points_2d[..., 1] * (y_max - y_min) + y_min  # y 范围
    return points_2d

def project_points(points_3d, extr_np, intr_np, img_size=(image_resolution, image_resolution)):
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

    N, _ = points_3d.shape

    # **1. NumPy 转 PyTorch**
    extr = torch.tensor(extr_np, dtype=torch.float32, device=points_3d.device)  # (4, 4)
    intr = torch.tensor(intr_np, dtype=torch.float32, device=points_3d.device)  # (3, 3)

    # **2. 扩展点为齐次坐标**
    ones = torch.ones((N, 1), device=points_3d.device)  # (N, 1)
    points_homo = torch.cat([points_3d, ones], dim=-1)  # (N, 4)

    # **3. 世界坐标 -> 相机坐标**
    R = np.array(extr_np[:3, :3], np.float32).T  # (3, 3) 变换矩阵
    T = np.array(extr_np[:3, 3], np.float32)  # (3,) 平移向量
    matrix = getWorld2View2(R, T)
    
    points_camera_homo = points_homo @ matrix.T  # (N, 4)
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

    return torch.stack([u, v], dim=-1)  # (N, 2)

def convert_image_to_camera_coords(points, image_width, image_height):
    """
    将 (B, N, 2) 形状的图像坐标系 (左上角原点) 转换为相机归一化坐标系 (中心为原点)
    
    参数：
        points: (B, N, 2) numpy 数组，每个点是 [u, v]，表示图像坐标
        image_width: 图像宽度 W
        image_height: 图像高度 H
    
    返回：
        transformed_points: (B, N, 2) numpy 数组，每个点是 [x, y]，表示相机坐标
    """
    # 拆分 u, v
    u, v = points[..., 0], points[..., 1]  # 形状: (B, N)
    
    # 变换
    x = u - (image_width / 2) + 0.5
    y = (image_height / 2) - v - 0.5  # 翻转 y 轴
    
    return np.stack([x, y], axis=-1)  # (B, N, 2)

def flip_y_axis(points_2d, img_size):
    """
    对 2D 点的 y 坐标进行翻转，使其关于图像的水平中轴线对称。
    
    参数:
    - points_2d: (B, N, 2) 的 Tensor，存储 (x, y) 坐标
    - img_size: (H, W) 图像尺寸
    
    返回:
    - 转换后的 (B, N, 2) Tensor
    """
    H = img_size  # 获取图像高度 H
    flipped_points = points_2d.clone()
    flipped_points[..., 1] = H - points_2d[..., 1]  # y' = H - y
    return flipped_points


# 生成二维点
B, N = 1, 1000
coordinate_bounds = [0, 0, image_resolution, image_resolution]
points_2d = generate_2d_points(B, N, coordinate_bounds)  # 左上角坐标系（左手系）



extr_np = np.array([
    [-0.863209, -0.213358, -0.457547, 0.842468],
    [-0.504848, 0.364807, 0.782333, -0.884787],
    [-0.000001, 0.906308, -0.422618, 1.579998],
    [0, 0, 0, 1]
])

# **内参 (NumPy)**
intr_np = np.array([
    [-110.851248, 0, 64],
    [0, -110.851248, 64],
    [0, 0, 1]
])


ray_dirs = init_ray_dirs()
points_2d_fipped = flip_y_axis(points_2d, image_resolution)   # 左下角坐标系（右手系）
filter_ray = filter_ray_(ray_dirs, points_2d_fipped)          # 相机坐标系（中心，右手系）
points_2d_cam = torch.tensor(convert_image_to_camera_coords(points_2d, image_resolution, image_resolution))


# 进行2d->3d投影
points_3d = []  # 世界坐标系
for i in range(B):
    depth = generate_depth(N)
    points_3d.append(single_batch_2d_to_3d(filter_ray[i], depth, extr_np))
points_3d = torch.cat(points_3d, dim=0)
# print(points_2d)               
# print(points_3d)

points_2d_project = []
for i in range(B):
    # 进行3d->2d投影
    points_2d_project.append(project_points(points_3d[i], extr_np, intr_np))
points_2d_project = torch.cat(points_2d_project, dim=0)
# print(points_2d_project)        # 左上角坐标系（左手系）


import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === 1. 相机参数 ===
extr_np = np.array([
    [-0.863209, -0.213358, -0.457547, 0.842468],
    [-0.504848,  0.364807,  0.782333, -0.884787],
    [-0.000001,  0.906308, -0.422618,  1.579998],
    [0, 0, 0, 1]
])
intr_np = np.array([
    [110.851248, 0, 64],
    [0, 110.851248, 64],
    [0, 0, 1]
])

# === 2. 你的 Tensor 转换成 NumPy ===
points_2d_project_torch = points_2d

points_3d_torch = points_3d

points_2d_project = points_2d_project_torch.numpy()[0]  # (10, 2)
points_3d = points_3d_torch.numpy()[0]  # (10, 3)

# === 3. 计算投影 2D 点的 3D 位置 ===
K_inv = np.linalg.inv(intr_np)  # 计算内参逆矩阵
points_2d_h = np.hstack((points_2d_project, np.ones((points_2d_project.shape[0], 1))))  # 齐次坐标
points_2d_cam = (K_inv @ points_2d_h.T).T  # 2D 变换到相机坐标系
points_2d_cam /= points_2d_cam[:, 2:]  # 归一化

# 变换到世界坐标系
camera_center = extr_np[:3, 3]  # 相机位置
points_2d_world = (extr_np[:3, :3] @ points_2d_cam.T).T + camera_center

# === 4. 绘制 3D 视图 ===
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制相机位置
ax.scatter(*camera_center, color='red', s=100, label="Camera Center")

# 绘制 3D 世界点
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], color='blue', s=50, label="3D Points")

# 绘制投影 2D 点
ax.scatter(points_2d_world[:, 0], points_2d_world[:, 1], points_2d_world[:, 2], color='green', s=50, label="Projected 2D Points")

# 画红色连线连接 3D 点和投影点
for p3d, p2d in zip(points_3d, points_2d_world):
    ax.plot([p3d[0], p2d[0]], [p3d[1], p2d[1]], [p3d[2], p2d[2]], color='red')
    
ax.view_init(elev=-20, azim=0)  # 设置视角

# 轴标签
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.legend()

plt.show()
