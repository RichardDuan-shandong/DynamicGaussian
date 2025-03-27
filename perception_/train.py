"""

    FileName          : train.py
    Author            : RichardDuan
    Last Update Date  : 2025-03-10
    Description       : to train the perception module, 
    Version           : 1.0
    License           : MIT License
    
"""
import pickle
from voxel.voxel_grid import VoxelGrid
import numpy as np
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
from termcolor import cprint
import lightning as L
import PIL.Image as Image
from perception_.mask_generator._2d_seg_generator import _2d_seg_generator
from pathlib import Path
from tqdm import tqdm
from perception_.gaussian_recon import SkeletonSplatter, GaussianSampler, GaussianRenderer, GaussianRecon
from perception_.gaussian_rendering.utils import save_multiview_image, get_merged_masks
from perception_.loss import l1_loss, l1_loss_mask, gmm_loss
import pytorch_msssim
from termcolor import cprint

def save_checkpoint(model, optimizer, epoch, filepath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def save_checkpoint(model, optimizer, epoch, filepath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")
           
def PSNR_torch(img1, img2, max_val=1):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = max_val
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
    
def check_and_make(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

'''
    format:
        data_ = data[i]
        1.data_[‘input_view’] : Dict
            ->data_[‘input_view’][‘image’]:Tensor 图片后续要走UNet进行特征预测
            ->data_[‘input_view’][‘pose’]:(Tensor,Tensor,Tensor)：这个后面可能用不到
              : camera_extrinsic = data_[‘input_view’][‘pose’][0], 
              : camera_intrinsic = data_[‘input_view’][‘pose’][1],
              : focal            = data_[‘input_view’][‘pose’][2].
              
            ->data_[‘input_view’][‘masks’] : List data[‘input_view’][‘masks’][0]:tensor 这个可以在上面做GMM获得2D分布

        2.data_[‘gt_view’] : List (len = multi_view_num) 
            ->data_[‘gt_view’][0][‘image’]:Tensor 图片后续要与predict做loss
            ->data_[‘gt_view’][0][‘pose’]:(Tensor,Tensor,Tensor) 这个在渲染时候会用
              : camera_extrinsic = data_[‘input_view’][‘pose’][0], 
              : camera_intrinsic = data_[‘input_view’][‘pose’][1],
              : focal            = data_[‘input_view’][‘pose’][2].
              
            ->data_[‘gt_view’][0][‘masks’]:Tensor 图片后续要与predict做局部重要性loss
        
        3.data_['task']: 任务标识，后续会作为表征输入进去      
'''
def _preprocess_data(seg_generator, data_save_path, batch_data, cfg, fabric): 
    data = []
    
    # batch_data is composed of several steps of data by multi_view
    for single_step_multiview in batch_data:
        
        task = single_step_multiview['description'][0]
        images_multiview = single_step_multiview['image']
        poses_multiview = single_step_multiview['pose']
        
        import torchvision.transforms.functional as F
        # images : tensor -> image
        images = []
        
        for image_tensor in images_multiview:  
            if image_tensor.shape[-1] in [1, 3, 4]:  # 确保通道数正确
                image_tensor = image_tensor.permute(2, 0, 1)  # (C, H, W)
            else:
                raise ValueError(f"Unexpected image shape: {image_tensor.shape}")
            image = np.asarray(image_tensor).astype(np.uint8)
            image = np.transpose(image, (1, 2, 0))
            image_pil = Image.fromarray(image)
            images.append(image_pil)
            
        # multi_view_sub_part_masks是在原图(128, 128)坐标系下的mask
        multi_view_sub_part_masks, multi_view_sub_part_images = seg_generator.seg_detail(images, task)
        debug = False
        # DEBUG
        if debug == True:
            import cv2
            import os
            images_list = [torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0)
               for img in images]
            for s in range(len(multi_view_sub_part_masks)):
                
                    image = images_list[cfg.train.image_choose[s]].clone()
                    single_image_masks = multi_view_sub_part_masks[s]
                    image = image.squeeze(0)
                    image = image.permute(1, 2, 0).numpy()
                    image = np.asarray(image).astype(np.uint8)
                #  print(image)
   
                    combined_mask = np.zeros((128,128), dtype=np.uint8)
                    m = 0
                    for single_object_masks in single_image_masks:
                        if len(single_object_masks) <= 0:
                            continue
                        for image_mask in single_object_masks:
                            m += 1

                            combined_mask = np.logical_or(combined_mask, image_mask).astype(np.uint8)
                                # 确保掩码是0和255的二值图像
                    # 应用掩码
                    masked_image = cv2.bitwise_and(image, image, mask=combined_mask)
                    # 将 numpy 数组转换为 PIL 图像对象
                    pil_image = Image.fromarray(masked_image)
                    mask_path = os.path.abspath(f'mask_view_{s}.png')
                    # 保存图像
                    pil_image.save(mask_path)
                
                    print(f"Masked image saved to  {mask_path} {m}")
                    
        for i in range(len(multi_view_sub_part_masks)):
            data_ = {}
            data_['input_view'] = {}
            data_['gt_view'] = []
            
            data_['input_view']['image'] = single_step_multiview['image'][cfg.train.image_choose[i]]
            data_['input_view']['pose'] = [single_step_multiview['pose'][0][cfg.train.image_choose[i]], single_step_multiview['pose'][1][cfg.train.image_choose[i]], single_step_multiview['pose'][2][cfg.train.image_choose[i]]]
            data_['input_view']['masks'] = multi_view_sub_part_masks[i]
            
            for j in range(len(multi_view_sub_part_masks)):
                if i != j:
                    gt_view = {}
                    gt_view['image'] = single_step_multiview['image'][cfg.train.image_choose[j]]
                    gt_view['pose'] =  [single_step_multiview['pose'][0][cfg.train.image_choose[j]], single_step_multiview['pose'][1][cfg.train.image_choose[j]], single_step_multiview['pose'][2][cfg.train.image_choose[j]]]
                    gt_view['masks'] = multi_view_sub_part_masks[j]
                    data_['gt_view'].append(gt_view)
            
            data_['task'] = task
                    
            data.append(data_)

    with open(data_save_path, 'wb') as f:
        pickle.dump(data, f)
        print('preprocessed data successfully saved!')

    return data

def check_tensor_status(tensor, name):
    """ 检查张量是否存在 NaN、Inf，并检测是否追踪梯度 """
    if torch.isnan(tensor).any():
        print(f"{name} contains NaN")
    if torch.isinf(tensor).any():
        print(f"{name} contains Inf")
    
    # 检查是否追踪梯度
    if tensor.requires_grad:
        print(f"{name} requires gradient.")
    else:
        print(f"{name} does NOT require gradient.")
        
def train(
        origin_data,
        cfg,
        fabric
    ):
    
    tasks = cfg.rlbench.tasks
    seg_generator = None
    data = None
    torch.cuda.empty_cache()

    # gaussian_recon = GaussianRecon(cfg, fabric)
    # image_predict, pcds, skeleton_gmm = gaussian_recon(data)    # (B,N,128,128,3)->[0, 1]
    gaussian_splatter = fabric.module(SkeletonSplatter(cfg, fabric))
    gaussian_sampler = fabric.module(GaussianSampler(cfg, fabric))
    gaussian_renderer = fabric.module(GaussianRenderer(cfg, fabric))
    
    optimizer_gaussian_splatter = torch.optim.Adam(gaussian_splatter.parameters(), lr=cfg.train.lr)
    optimizer_gaussian_renderer = torch.optim.Adam(gaussian_renderer.parameters(), lr=cfg.train.lr)
    
    start_epoch = 0
    step = 0
    
    for epoch in range(start_epoch, cfg.train.epochs):
        for task in tasks:  # 依次遍历每个任务场景
            single_task_data = origin_data[task]
            index = 0
            # 每一个batch都要加载
            for batch_data in single_task_data: # 取出每个任务的data，按一个小batch进行处理
                # ==0. get preprocess data==
                # check if the data has been loaded
                data_save_file = os.path.join(cfg.train.seg_save, task)
                check_and_make(data_save_file)
                data_save_path = os.path.join(data_save_file, f'{index}.pkl')
                reload_flag = False
                if os.path.exists(data_save_path):  # 如果发现本条数据保存了，就直接读取就可以，不用走preprocess
                    with open(data_save_path, 'rb') as f:
                        # print(data_save_path)
                        data = pickle.load(f)
                    reload_flag = True
                    # 确保从外部加载的data符合数据格式
                    if data is None or len(data) != cfg.train.batch_size * len(cfg.train.image_choose):
                        print("read in data format is invalid, will reload it.")
                        reload_flag = False
                # 如果没有预先保存或者保存的不符合数据形式则重新加载
                if reload_flag != True:
                    if seg_generator is None:
                        seg_generator = _2d_seg_generator(cfg, fabric)
                    data = _preprocess_data(seg_generator, data_save_path, batch_data, cfg, fabric)
                index += 1
                # print(data)
                
                # ==1. get ground truth image and mani-related object mask==
                image_groundtruth = []
                image_groundtruth_mask = []
                for i in range(len(data)):  # 遍历每个batch
                    single_batch_data_gt_view = []
                    single_batch_data_gt_mask = []
                    for j in range(len(data[i]['gt_view'])):    # 遍历每个视角
                        single_batch_data_gt_view.append((data[i]['gt_view'][j]['image'].float() / 255.0).unsqueeze(0))
                        single_batch_data_gt_mask.append(get_merged_masks(data[i]['gt_view'][j]['masks'], data[i]['gt_view'][j]['image']).unsqueeze(0))
                    single_batch_gt_view = torch.cat(single_batch_data_gt_view, dim=0)     # (1, N, 128, 128, 3)
                    single_batch_data_gt_mask = torch.cat(single_batch_data_gt_mask, dim=0).unsqueeze(0)      # (1, N, 128, 128)
                    image_groundtruth.append(single_batch_gt_view)
                    image_groundtruth_mask.append(single_batch_data_gt_mask)
                image_groundtruth = torch.stack(image_groundtruth, dim=0)  # (B, N, 128, 128, 3)
                image_groundtruth_mask = torch.cat(image_groundtruth_mask, dim=0) # (B, N, 128, 128)
                # # debug
                # save_multiview_image(image_groundtruth, "gt")
                # save_multiview_image(image_predict, "predict")
                
                
                # ==2. get reconstructed data(with single view input and multiview output)==
                
                # get 3d gaussian skeleton through a depth_predictor
                means_3d, cov_3d, weight_3d, object_index, feature_map = gaussian_splatter(data)
                # sample the 3d gaussian skeleton to get pcds
                pcds, pcds_project_to_image, skeleton_gmm = gaussian_sampler.sample(data=data, means_3d=means_3d, cov_3d=cov_3d, weight_3d=weight_3d, object_index=object_index)
                
                for inner_loop in range(cfg.train.inner_loop_epochs):
                    
                    # decode the pcds to get a gaussian splatting pcds, and render the predicted_image
                    image_predict, new_pcds, gaussian_params_list, debug_loss = gaussian_renderer(data, pcds, pcds_project_to_image, feature_map)
                    # calculate rendering_loss
                    image_groundtruth = image_groundtruth.to(device=image_predict.device)
                    image_groundtruth_mask = image_groundtruth_mask.to(device=image_predict.device)
                    image_loss = l1_loss(image_predict, image_groundtruth)
                    mask_loss = l1_loss_mask(image_predict, image_groundtruth, image_groundtruth_mask)
                    ssim_loss = pytorch_msssim.ssim(image_predict, image_groundtruth, data_range=1)
                    lamada_image_loss = cfg.train.lamada_image_loss
                    lamada_mask_loss = cfg.train.lamada_mask_loss
                    lamada_ssim_loss = cfg.train.lamada_ssim_loss
                    rendering_loss = lamada_image_loss * image_loss + lamada_mask_loss * mask_loss + lamada_ssim_loss * (1 - ssim_loss) 
                   
                    print("---------------------------------------------------")
                    print("debug_loss", debug_loss)
                    optimizer_gaussian_renderer.zero_grad()
                    debug_loss.backward()
                    optimizer_gaussian_renderer.step()

                    print("---------------------------------------------------")

                    
                    optimizer_gaussian_decoder.zero_grad()
                    rendering_loss.backward()
                    optimizer_gaussian_decoder.step()
                    
                    pcds = new_pcds
                    
                    step+=1
                    print("================================================")
                gmm_consistency_loss = 0
                for t in range(len(pcds)):
                    _pcds = pcds[t]
                    _skeleton_gmm = skeleton_gmm[t]
                    gmm_consistency_loss += gmm_loss(points=_pcds, means=_skeleton_gmm['means'], covariances=_skeleton_gmm['cov'],  weights=_skeleton_gmm['weight'])


                lamada_skeleton_loss = cfg.train.lamada_skeleton_loss
                # loss =  lamada_skeleton_loss * gmm_consistency_loss
                skeleton_consistency_loss = lamada_skeleton_loss * gmm_consistency_loss
                
                optimizer_gaussian_splatter.zero_grad()
                gmm_consistency_loss.backward()
                optimizer_gaussian_splatter.step()

                cprint(f"step:{step} | image_loss:{image_loss}*{lamada_image_loss} | mask_loss:{mask_loss}*{lamada_mask_loss} | ssim_loss:{1-ssim_loss}*{lamada_ssim_loss} | gmm_loss:{gmm_consistency_loss}*{lamada_skeleton_loss}", 'green')
                save_multiview_image(image_predict, "predict")
                

        # 在每个 epoch 结束后保存检查点
        save_checkpoint(gaussian_splatter, optimizer_gaussian_splatter, epoch, f'checkpoint_gaussian_splatter_epoch_{epoch}.pth')
        save_checkpoint(gaussian_decoder, optimizer_gaussian_decoder, epoch, f'checkpoint_gaussian_decoder_epoch_{epoch}.pth')
    return