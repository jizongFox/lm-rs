import torch
from scene.optim_strategy.adamOptimizer import AdamOptimizer

from gaussian_renderer import render
import time

def adam_step(camera_sampler, iteration, debug_from, opt, scene, viewpoint_stack, gaussians, pipe, background, optimizerHandler:AdamOptimizer, batch_size):
    start_time = time.time()
    cam_time, render_time, backward_time, update_time = 0,0,0,0
    for current_batch in range(batch_size):
        
        viewpoint_cam = camera_sampler.get_camera(current_batch)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        gt_image = viewpoint_cam.original_image.cuda()

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        loss = optimizerHandler.get_shift(image, gt_image, gaussians)

    # optimizerHandler.step(gaussians)
    torch.cuda.synchronize()

    end_time = time.time()

    return loss, end_time - start_time, viewspace_point_tensor, visibility_filter, radii 