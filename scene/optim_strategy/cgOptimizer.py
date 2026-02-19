import torch
import matplotlib.pyplot as plt
import os
from arguments import GaussNewtonOptimizationParams
import numpy as np
from .residuals.get_residual import mse_residual, SimpleSSIMDerivative
from typing import NamedTuple
from .solvers import get_solver
from diff_gaussian_rasterization import _RasterizeGaussians

C0 = 0.28209479177387814
MAX_COLOR = 0.5 / C0
MIN_COLOR = -MAX_COLOR
GAUSSIAN_ATTRIBUTES = 14


class ResidualState(NamedTuple):
    mse_residuals: torch.tensor
    ssim_residuals: torch.tensor
    ssim_derivatives: torch.tensor


class CGOptimizer:

    def __init__(
        self,
        gaussians,
        gn_opt_params: GaussNewtonOptimizationParams,
        path,
        cameras_extend,
    ) -> None:
        self.gn_opt_params = gn_opt_params
        self.iteration = 1
        self.path = path

        self.fixed_lr = gn_opt_params.fixed_lr

        W, H, CHANNELS = gaussians.cgState.width, gaussians.cgState.height, 3
        self.batch_size = gn_opt_params.batch_size

        # Initialize Residual State
        deltaY_batch = torch.zeros((self.batch_size, W * H * CHANNELS), device="cuda")
        ssim_residuals = torch.zeros((self.batch_size, W * H * CHANNELS), device="cuda")
        delSSIM_delPixel = torch.zeros(
            (self.batch_size, W * H * CHANNELS), device="cuda"
        )
        self.residual_state = ResidualState(
            deltaY_batch, ssim_residuals, delSSIM_delPixel
        )

        self.ssim_derivative = SimpleSSIMDerivative()
        self.W, self.H, self.CHANNELS = W, H, CHANNELS
        self.losses = torch.zeros((self.batch_size))

        self.optim_iter = 1
        self.regularizer = gn_opt_params.regularizer
        self.max_lr = torch.tensor(gn_opt_params.max_lr, device="cuda")
        self.auto_lr = gn_opt_params.auto_lr
        self.lrs = []
        self.cameras_extend = cameras_extend
        solver_class = get_solver("CG")
        self.solver = solver_class(
            gaussians._xyz.shape[0] * GAUSSIAN_ATTRIBUTES,
            lambda_reg=gn_opt_params.regularizer,
            levenberg_type=gn_opt_params.levenberg_type,
        )

    def append_residual(self, image, gt_image, gaussians, current_batch):
        with torch.no_grad():
            deltaY, loss_mse = mse_residual(
                gt_image, image, gaussians.cgState.lambda_dssim
            )
            self.residual_state.mse_residuals[current_batch] = torch.permute(
                deltaY, (1, 2, 0)
            ).flatten()

            loss_ssim = 0
            ssim_residuals = 0
            if gaussians.cgState.loss_fn == "mse_approx_ssim":
                ssim_residuals, delSSIM_delPixel, loss_ssim = (
                    self.ssim_derivative.get_ssim_residuals(
                        image, gt_image, gaussians.cgState.lambda_dssim
                    )
                )

                self.residual_state.ssim_residuals[current_batch] = (
                    ssim_residuals.flatten()
                )
                self.residual_state.ssim_derivatives[current_batch] = delSSIM_delPixel

            self.losses[current_batch] = loss_mse + loss_ssim
        return deltaY

    def sched_update(self, key, new_value):
        # TODO: Update schedulers
        if key == "batchsize_sched" and self.batch_size != new_value:
            self.batch_size = new_value
            deltaY_batch = torch.zeros(
                (self.batch_size, self.W * self.H * self.CHANNELS), device="cuda"
            )
            ssim_residuals = torch.zeros(
                (self.batch_size, self.W * self.H * self.CHANNELS), device="cuda"
            )
            delSSIM_delPixel = torch.zeros(
                (self.batch_size, self.W * self.H * self.CHANNELS), device="cuda"
            )
            self.residual_state = ResidualState(
                deltaY_batch, ssim_residuals, delSSIM_delPixel
            )
            self.losses = torch.zeros((self.batch_size))

        if key == "cgiter_sched":
            self.solver.set_linear_iter(new_value)

    def linear_solve(self, gaussians, return_matvec_kernels):
        _RasterizeGaussians.set_residual_state(
            self.residual_state, gaussians.cgState.lambda_dssim
        )
        _RasterizeGaussians.set_backward_inputs(gaussians.cgState)
        self.solution = self.solver.linear_solve(debug=return_matvec_kernels)

        return self.losses.mean()

    def step(self, gaussians, line_search_lr=None):
        with torch.no_grad():
            # beta = self.solution.view(-1, 14)
            number_of_gaussians = gaussians._xyz.shape[0]
            param_names = ["opacity", "dc", "xyz", "scale", "rotation"]
            param_dims = [1, 3, 3, 3, 4]
            slices = []
            start_idx = 0
            for i in range(len(param_names)):
                dim = param_dims[i]
                slices.append(slice(start_idx, start_idx + dim * number_of_gaussians))
                start_idx = start_idx + dim * number_of_gaussians
            param_list = [
                gaussians._opacity,
                gaussians._features_dc,
                gaussians._xyz,
                gaussians._scaling,
                gaussians._rotation,
            ]

            if line_search_lr != None:
                lr = line_search_lr
            else:
                lr = self.fixed_lr
                max_lr = (
                    torch.tensor(0.05, device="cuda")
                    if self.optim_iter < 10
                    else self.max_lr
                )
                if self.auto_lr:
                    color_update = torch.abs(self.solution[slices[1]])
                    lr = 1 / color_update.max()
                    lr = torch.minimum(max_lr, lr)

            for i in range(len(param_list)):
                dim = param_dims[i]
                p = self.solution[slices[i]]
                p = p.view(dim, number_of_gaussians).T
                p = p.view_as(param_list[i])
                param_list[i] += p * lr
                if param_names[i] == "dc":
                    param_list[i].clamp_(min=MIN_COLOR)
                if param_names[i] == "scale":
                    param_list[i].clamp_(max=self.cameras_extend)

            scales = gaussians._scaling
            too_big = scales.max() > 15
            if too_big:
                print(
                    f"Iteration {self.optim_iter}: Scale is Getting to Infinity, Exiting..."
                )
                exit(1)
            self.lrs.append(lr)
            self.optim_iter = self.optim_iter + 1

    def plot_lr(self):
        lr = [
            item.item() if isinstance(item, torch.Tensor) else item for item in self.lrs
        ]
        # Create figure and axis objects using object-oriented style
        fig, ax = plt.subplots()

        # Plot the data
        ax.plot(
            lr,
            marker=".",
            markersize=4,
            linestyle="-",
            color="b",
            label="Learning Rate",
        )

        ax.set_title("Learning Rate vs Iteration")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("LR")
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

        print("Learning Rate is ", lr[-1])
        fig.savefig(os.path.join(self.path, "lr.png"), bbox_inches="tight")
        plt.close(fig)
        np.save(os.path.join(self.path, "lr.npy"), lr)
