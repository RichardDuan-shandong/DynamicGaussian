
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import einops
import torch.distributions as dist

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def l1_loss_mask(network_output, gt, mask):
    # 确保维度一致
    if mask.shape != network_output.shape:
        mask = mask.unsqueeze(-1)  # 将 mask 扩展到 (B, N, H, W, 1)

    # 计算 masked L1 loss
    loss = torch.abs(network_output - gt) * mask
    loss = loss.sum() / (mask.sum() + 1e-6)  # 避免除以0
    return loss

def gmm_loss(points, means, covariances, weights):
    """
    计算点云与GMM的负对数似然损失
    Args:
        points: (N, 3) 点云
        weights: (K,) 高斯混合权重（π）
        means: (K, 3) 高斯分布的均值
        covariances: (K, 3, 3) 高斯分布的协方差矩阵

    Returns:
        nll_loss: 负对数似然损失
    """
    N, _ = points.shape
    K = weights.shape[0]

    # 计算GMM的概率密度
    gmm_density = torch.zeros(N, device=points.device)

    for k in range(K):
        # 多元高斯分布
        mvn = dist.MultivariateNormal(means[k], covariance_matrix=covariances[k])
        # 计算每个点在当前高斯分布下的概率密度
        prob = mvn.log_prob(points).exp()  # 转为概率密度
        gmm_density += weights[k] * prob

    # 避免数值问题，加上一个小常数
    eps = 1e-8
    gmm_density = torch.clamp(gmm_density, min=eps)

    # 计算负对数似然损失
    nll_loss = -torch.log(gmm_density).mean()

    return nll_loss

def cosine_loss(network_output, gt):
    '''
    network_output: [B, H, W, C]
    gt: [B, H, W, C]
    '''
    return 1 - F.cosine_similarity(network_output, gt, dim=-1).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
