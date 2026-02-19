import torch
from scene.optim_strategy.cgOptimizer import CGOptimizer
from arguments import GaussNewtonOptimizationParams
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
import time


@torch.no_grad()
def post_render_task(
    pixel_sampler, iteration, cgState, current_batch, loss_map, likelihood_viz_freq
):
    # start_time = time.time()
    is_visualize = (
        likelihood_viz_freq > 0
        and current_batch == 0
        and (iteration - 1) % likelihood_viz_freq == 0
    )
    sample_size = cgState.sample_per_block

    if sample_size == 256:  # No Sampling
        sampled_pixels = cgState.full_image
        cgState.state["sampled_pixels"][current_batch] = sampled_pixels
        cgState.state["likelihoods"][current_batch][:] = 1 / 256
    else:
        pixel_sampler.sample(
            current_batch, cgState, visualize=is_visualize, loss_map=loss_map
        )


def gauss_newton_step(
    pixel_sampler,
    camera_sampler,
    iteration,
    debug_from,
    opt,
    gaussians: GaussianModel,
    pipe,
    background,
    optimizerHandler: CGOptimizer,
    gauss_newton_opts: GaussNewtonOptimizationParams,
):

    if (iteration - 1) == debug_from:
        pipe.enable_timer = True
    start_time = time.time()
    bg = torch.rand((3), device="cuda") if opt.random_background else background
    for current_batch in range(gaussians.cgState.batch_size):
        viewpoint_cam = camera_sampler.get_camera(current_batch)
        gt_image = viewpoint_cam.original_image.cuda()
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            pipe,
            bg,
            isbatched=True,
            end_transmittance=gauss_newton_opts.end_transmittance,
            current_batch=current_batch,
        )
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        gaussians.batchState.insert(current_batch, radii)
        loss_map = optimizerHandler.append_residual(
            image, gt_image, gaussians, current_batch
        )
        post_render_task(
            pixel_sampler,
            iteration,
            gaussians.cgState,
            current_batch,
            loss_map,
            gauss_newton_opts.likelihood_viz_freq,
        )

    loss = optimizerHandler.linear_solve(gaussians, pipe.return_matvec_kernels)
    torch.cuda.synchronize()
    end_time = time.time()
    return loss.item(), end_time - start_time
