import torch
from utils.loss_utils import l2_loss, l1_loss, ssim
from utils.simplified_ssim import simple_ssim_loss


class AdamOptimizer:
    def __init__(self, params, loss_fn) -> None:
        self.lambda_dssim = params.lambda_dssim
        self.lambda_scale = params.lambda_scale
        self.loss_fn = loss_fn

    def get_shift(self, image, gt_image, gaussians):
        if self.loss_fn == "mse":
            loss = l2_loss(image, gt_image, "mean")
        elif self.loss_fn == "mae":
            loss = l1_loss(image, gt_image)
        elif self.loss_fn == "mae_ssim":
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (
                1.0 - ssim(image, gt_image)
            )
        elif self.loss_fn == "mse_ssim":
            Ll2 = l2_loss(image, gt_image)
            ssim_value = ssim(image, gt_image)
            loss = Ll2 + self.lambda_dssim * ((1.0 - ssim_value) ** 2)
        elif self.loss_fn == "mse_approx_ssim":  # Same as our second-order optimizer
            Ll2 = l2_loss(image, gt_image)
            ssim_loss = simple_ssim_loss(image, gt_image).mean()
            loss = Ll2 + self.lambda_dssim * (1.0 - ssim_loss)

        loss.backward()
        return loss.item()

    def step(self, gaussians):
        with torch.no_grad():
            gaussians.optimizer.step()
            # gaussians._scaling = torch.clamp(gaussians._scaling, max=self.scale_cutoff)
            gaussians.optimizer.zero_grad(set_to_none=True)
