import torch
import numpy as np
from gaussian_renderer import render
import time
from random import randint
from utils.loss_utils import l2_loss
import os

def get_jacobian(image, x):
    """
    Each tuple shape: 10000 x n
    n is 1 for opacity, 3 for color, 3 for mean, 3 scale and 4 for rotation
    Concat creates a matrix with 10000 x 14
    Flatten creates a 1D array, where every 14 element is associated with a single Gaussian
    """
    image = image.flatten()
    ones = torch.zeros_like(image)
    jacobian_rows=[]

    for i in range( ones.shape[0]):
        ones[i] = 1.0
        ones[i-1] = 0.0
        gradient_tuple = torch.autograd.grad( image, x, ones, retain_graph=True)
        jacobian_row = torch.cat(tuple(map( lambda t: t.view(t.shape[0], -1), gradient_tuple)), dim=1).flatten()
        jacobian_rows.append(jacobian_row)

    jacobian_rows = torch.stack(jacobian_rows)
    jacobian_rows = jacobian_rows.view(jacobian_rows.shape[0], -1)
    return jacobian_rows.detach()


def test_diag(image, x) -> torch.Tensor:
    image = image.flatten()
    ones = torch.zeros_like(image)
    diag = torch.zeros( (x[0].shape[0] * 14, ), device = "cuda")
    for i in range( ones.shape[0]):
        ones[i] = 1.0
        ones[i-1] = 0.0
        gradient_tuple = torch.autograd.grad( image, x, ones, retain_graph=True)
        # jacobian_row = torch.cat(  tuple(map( lambda x: x.flatten(), gradient_tuple)))
        
        jacobian_row = torch.cat(tuple(map( lambda t: t.view(t.shape[0], -1), gradient_tuple)), dim=1).flatten()
        diag = diag + (jacobian_row ** 2)
        if (i % 100000 == 0):
            print(f"{i} / { ones.shape[0]}")

    return diag

def get_gradient(gaussians, shp) -> torch.Tensor:
    with torch.no_grad():
        grad_vec = torch.cat((gaussians._opacity.grad, gaussians._features_dc.grad.squeeze(1), gaussians._xyz.grad ,gaussians._scaling.grad, gaussians._rotation.grad), dim=1)
        grad_vec = grad_vec.flatten() * shp[0] * shp[1] * shp[2] / 2
        return -grad_vec

def save_jacob(batch_size, scene, viewpoint_stack, gaussians, pipe, background, camera_sampler, model_path):
    path = os.path.join(model_path, "Jacobians")

    start = time.time()
    for current_batch in range(batch_size):
        viewpoint_cam = camera_sampler.get_camera(current_batch)
        gt_image = viewpoint_cam.original_image.cuda()

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        param_pack = [ gaussians._opacity, gaussians._features_dc, gaussians._xyz, gaussians._scaling, gaussians._rotation]
        image = torch.permute(image, (1,2,0))
        J = get_jacobian(image, param_pack)
        with torch.no_grad():
            J = J.cpu().numpy()
            np.savez_compressed( os.path.join(path, f"Jacobian_batch{current_batch}.npz"), J)
            residual = (torch.permute(gt_image , (1, 2, 0)) - image).flatten()
            np.savez_compressed(os.path.join(path, f"residual_batch{current_batch}.npz"), residual.cpu().numpy())
            print(f"Saved Batch {current_batch}!")
            del J
    
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print("All Saved!")
    print("Time elapsed: ", elapsed / 60, " Minutes")
    print("Memory Required: ", torch.cuda.max_memory_allocated() * 1e-9, " GB")
    exit(0)
    return None, None





