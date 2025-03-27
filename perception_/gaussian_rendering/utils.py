import torch
import numpy as np
import torch.autograd.profiler as profiler
import visdom
import time
from lightning.fabric import Fabric


# REWARD_SCALE = 100.0
# LOW_DIM_SIZE = 4
@torch.no_grad()
def visualize_pcd(xyz, attention_coordinate=None, rgb=None, name='xyz', sleep=0):
    '''
    use visdom to visualize point cloud in training process
    xyz: (B, N, 3)
    rgb: (B, 3, H, W)
    '''
    vis = visdom.Visdom()
    if rgb is not None:
        rgb_vis = rgb[0].detach().cpu().numpy()
        vis.image(rgb_vis, win='rgb', opts=dict(title='rgb'))

    # point cloud
    pc_vis = xyz[0].detach().cpu().numpy()  # (128*128, 3)

    # visualize ground-truth action_trans (B,3) in point cloud (blue)
    if attention_coordinate is not None:
        action = attention_coordinate[0].unsqueeze(0).detach().cpu().numpy()
        pc_vis_aug = np.concatenate([pc_vis, action], axis=0)
        label_vis = np.concatenate([np.zeros((pc_vis.shape[0], 1))+1, np.zeros((1,1))+2], axis=0)
    else:
        pc_vis_aug = pc_vis
        label_vis = np.zeros((pc_vis.shape[0], 1))+1
    label_vis = label_vis.astype(int)

    vis.scatter(
        X=pc_vis_aug, Y=label_vis, win=name, 
        opts=dict(
            title=name,
            markersize=1,
            markercolor=np.array([[0,0,255], [255,0,0]]) if attention_coordinate is not None else np.array([[0,0,255]]),
            # blue and red
        )
    )
    if sleep > 0:
        time.sleep(sleep)
@torch.no_grad()
def get_merged_masks(single_image_mask, image):
    '''
    param:
    masks:   ->_data['gt_view'][i]['masks']
    return 
    '''
    import cv2
    from PIL import Image
    import os
    combined_mask = np.zeros((128,128), dtype=np.uint8)
    for single_object_masks in single_image_mask:
        if len(single_object_masks) <= 0:
            continue
        for mask in single_object_masks:
            combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
    
    # image = (image.numpy()).astype(np.uint8)
    # print(image.shape)
    # masked_image = cv2.bitwise_and(image, image, mask=combined_mask)
    # pil_image = Image.fromarray(masked_image)
    # mask_path = os.path.abspath(f'mask.png')
    # pil_image.save(mask_path)
    
    return torch.tensor(combined_mask)
@torch.no_grad()
def save_multiview_image(images:torch.Tensor, tag=None):
    '''
    param:
    images:(B,N,H,W,C)->(10,4,128,128,3)
    '''
    
    B, N, _, _, _ = images.shape
    import os
    from PIL import Image
    for i in range(B):
        single_batch_image = images[i].squeeze(0)
        for j in range(N):
            image = single_batch_image[j].squeeze(0).clone()

            image = image.cpu().numpy() * 255
            image = np.asarray(image).astype(np.uint8)
            pil_image = Image.fromarray(image)
            if tag is not None:
                image_path = os.path.abspath(f'multi_view_{j}_{tag}.png')
               # image_path = os.path.abspath(f'batch_{i}_multi_view_{j}_{tag}.png')
            else:
                image_path = os.path.abspath(f'multi_view_{j}.png')
               # image_path = os.path.abspath(f'batch_{i}_multi_view_{j}.png')
            pil_image.save(image_path)
            # print(f"Image saved to {image_path}")


class PositionalEncoding(torch.nn.Module):
    """
    Implement NeRF's positional encoding
    """

    def __init__(self, num_freqs=6, d_in=3, freq_factor=np.pi, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (batch, self.d_in)
        :return (batch, self.d_out)
        """
        with profiler.record_function("positional_enc"):
            embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
            embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
            embed = embed.view(x.shape[0], -1)
            if self.include_input:
                embed = torch.cat((x, embed), dim=-1)
            return embed

    @classmethod
    def from_conf(cls, conf, d_in=3):
        # PyHocon construction
        return cls(
            conf.num_freqs,
            d_in,
            conf.freq_factor,
            conf.include_input,
        )
        