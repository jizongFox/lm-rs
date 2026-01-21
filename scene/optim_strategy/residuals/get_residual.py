# Implementation follows Loss Functions for "Image Restoration with Neural Networks" paper
# It is also inspired from the preprint of 3DGS-LM

import torch
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
import math

def mse_residual(gt_image, image, lambda_ssim=0, return_only_loss=False):
    # return (1.0 - lambda_ssim) * (torch.permute((gt_image - image), (1, 2, 0)).flatten())
    residual =  (gt_image - image) #torch.permute((gt_image - image), (1, 2, 0)).flatten()
    loss = torch.sum(residual**2) # sum of squared loss
    if return_only_loss:
        return loss
    return residual, loss


class SimpleSSIMDerivative:
    def __init__(self, channel=3, window_size=11) -> None:
        self.channel = channel
        self.window_size = window_size
        self.window = self.create_window(self.window_size, self.channel).cuda()
        self.GPeak = 0.07076223194599152

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window
    
    def ssim_forward(self, img1, img2, window, window_size, channel):
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

        luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
        contrast_structure =  (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        ssim_map = luminance * contrast_structure

        return {"ssim_map": ssim_map, "luminance":luminance,
                "contrast_structure": contrast_structure,
                "mu1": mu1, "mu2": mu2, 
                "mu1_sq": mu1_sq, "mu2_sq": mu2_sq,
                "sigma1_sq" : sigma1_sq, "sigma2_sq" : sigma2_sq}
    

    def ssim_simple_backward(self, img1, img2, backward_dict):
        luminance = backward_dict["luminance"]
        contrast_structure = backward_dict["contrast_structure"]
        mu1 = backward_dict["mu1"]
        mu2 = backward_dict["mu2"]
        mu1_sq = backward_dict["mu1_sq"]
        mu2_sq = backward_dict["mu2_sq"]
        sigma1_sq = backward_dict["sigma1_sq"]
        sigma2_sq = backward_dict["sigma2_sq"]


        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        delLuminance_delPixel = 2 * self.GPeak * ( (mu2 - mu1 * luminance) / (mu1_sq + mu2_sq + C1))
        delContrastStructure_delPixel = (img2 - mu2) - contrast_structure * (img1 - mu1)
        delContrastStructure_delPixel = (2 * self.GPeak * delContrastStructure_delPixel) / (sigma1_sq + sigma2_sq + C2)

        delSSIM_delPixel = delLuminance_delPixel * contrast_structure + luminance * delContrastStructure_delPixel
        return delSSIM_delPixel

    def get_ssim_residuals(self, rendered, gt, lambda_ssim, return_only_loss=False):
        backward_dict = self.ssim_forward(rendered, gt, self.window, self.window_size, self.channel)
        ssim_residuals = 1 - backward_dict["ssim_map"]
        ssim_residuals = math.sqrt(lambda_ssim) * (ssim_residuals)
        loss = torch.sum(ssim_residuals**2) # Squred loss
        if return_only_loss:
            return loss
        
        delSSIM_delPixel = self.ssim_simple_backward(rendered, gt, backward_dict)
        delSSIM_delPixel = math.sqrt(lambda_ssim) * (delSSIM_delPixel.flatten())

        # ssim_residuals_sqrt = torch.sqrt(ssim_residuals).flatten() # sqrt(1 - SSIM(i))
        # ssim_residuals = lambda_ssim * ssim_residuals_sqrt
        # delSSIM_delPixel = lambda_ssim * 0.5 * (delSSIM_delPixel.flatten()) / ssim_residuals_sqrt


        return ssim_residuals, delSSIM_delPixel, loss
