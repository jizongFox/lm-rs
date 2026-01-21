import torch
import torch.nn.functional as F
from torch.autograd import Function
from math import exp
from torch.autograd import Variable


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

class SimpleSSIMFunction(Function):
    @staticmethod
    def forward(ctx, img1, img2, window, window_size, channel, GPeak):
        # Compute local means
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Compute local variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Compute SSIM components
        luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
        contrast_structure = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        ssim_map = luminance * contrast_structure

        # Save all necessary tensors for backward
        ctx.save_for_backward(img1, img2, mu1, mu2, mu1_sq, mu2_sq, sigma1_sq, sigma2_sq)
        ctx.window = window
        ctx.window_size = window_size
        ctx.channel = channel
        ctx.luminance = luminance
        ctx.contrast_structure = contrast_structure
        ctx.GPeak = GPeak
        ctx.C1 = C1
        ctx.C2 = C2

        return ssim_map

    @staticmethod
    def backward(ctx, grad_output):
        img1, img2, mu1, mu2, mu1_sq, mu2_sq, sigma1_sq, sigma2_sq = ctx.saved_tensors
        window_size = ctx.window_size
        channel = ctx.channel
        luminance = ctx.luminance
        contrast_structure = ctx.contrast_structure
        GPeak = ctx.GPeak
        C1 = ctx.C1
        C2 = ctx.C2

        # Compute derivative of luminance term w.r.t. pixels of img1
        delLuminance_delPixel = 2 * GPeak * ((mu2 - mu1 * luminance) / (mu1_sq + mu2_sq + C1))

        d_contrast_structure = (img2 - mu2) - contrast_structure * (img1 - mu1)
        delContrastStructure_delPixel = (2 * GPeak * d_contrast_structure) / (sigma1_sq + sigma2_sq + C2)

        delSSIM_delPixel = delLuminance_delPixel * contrast_structure + luminance * delContrastStructure_delPixel

        grad_img1 = grad_output * delSSIM_delPixel

        grad_img2 = None

        return grad_img1, grad_img2, None, None, None, None

def simple_ssim_loss(rendered, gt, lambda_ssim=1.0, channel=3, window_size=11):
    """
    Compute the simplified SSIM loss between rendered and ground-truth images.
    """
    # Create the Gaussian window once (you might cache this if the parameters do not change)
    window = create_window(window_size, channel).to(rendered.device)
    GPeak = 0.07076223194599152

    ssim_map = SimpleSSIMFunction.apply(rendered, gt, window, window_size, channel, GPeak)
    return ssim_map

if __name__ == '__main__':
    rendered = torch.rand(1, 3, 64, 64, requires_grad=True).cuda()
    gt = torch.rand(1, 3, 64, 64).cuda()
    
    loss = simple_ssim_loss(rendered, gt, lambda_ssim=1.0).mean()
    loss.backward()
    print("Loss:", loss.item())
