import numpy as np
import os
import argparse
import math

"""
Jacobian Layout (Pixels*Channels) * (Total Gaussians * 14)
The layout: 
        FirstGaussian params: o c1 c2 c3 x1 x2 x3 s1 s2 s3 r1 r2 r3 r4  SecondGaussian params ....
Pixel1
r
g
b
Pixel2
.
.
.

Residual follows the same pixel format

CUDA Layout (Gradient, Diagonal and JTJv).

First All the Opacity Params: o_1 o_2 o_3 .... o_{total gaussians} Then the first color parameter c1_1 c1_2 .... c1_{total_gaussians} 
"""

attribute_to_name = {0: "opacity", 
                     1: "color_r", 
                     2: "color_g", 
                     3: "color_b", 
                     4: "mean_x", 
                     5: "mean_y", 
                     6: "mean_z",
                     7: "scale_x", 
                     8: "scale_y", 
                     9: "scale_z", 
                     10: "rotation_r", 
                     11: "rotation_x", 
                     12: "rotation_y", 
                     13: "rotation_z"}
parser = argparse.ArgumentParser(
        description="Tests the CUDA results."
    )
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--Jpath", type=str, 
    default="tests/GT_Jacobians/Jacobians", help="Main path to GT Jacobians")
parser.add_argument("--J_cuda_path", type=str, 
    default="tests/gauss-newton-ckpt/cuda_results", help="Main path to CUDA results")
parser.add_argument("--atol", type=float, 
    default=1e-5, help="Absolute Tolerance for Comparisons")
parser.add_argument("--rtol", type=float, 
    default=1e-3, help="Relative Tolerance for Comparisons")
parser.add_argument("--rtol_jtjv", type=float, 
    default=1e-1, help="Relative Tolerance for JTJv Product Comparisons")
parser.add_argument("--failure_threshold", type=float, 
    default=0.1, help="Numerical errors occur in computations. At what percentage we should be extra careful?")

args = parser.parse_args()

def info(name, cuda, gt ):
    print(f"------------------[WARNING] {name} Computation Has Errors")
    diff_idx = np.isclose(cuda, gt, atol=args.atol, rtol=args.rtol) == False
    print(f"GT {name} ", gt[diff_idx])
    print(f"CUDA {name} ", cuda[diff_idx])
    incorret_idx = np.where(diff_idx)[0]
    total_gaussians = gt.shape[0] // 14
    incorrect_gaussians = incorret_idx % total_gaussians
    incorrect_attributes = incorret_idx // total_gaussians
    print(f"Following indices are calculated incorrectly: {incorret_idx}")
    print(f"Following Gaussians are calculated incorrectly: {incorrect_gaussians}")
    attribute_names = [attribute_to_name[key] for key in incorrect_attributes]
    print(f"Following Attributes are calculated incorrectly: {attribute_names}")
    wrong_count = np.count_nonzero(diff_idx)
    total = np.count_nonzero(gt)
    failure_rate = wrong_count / total * 100
    print("Number of mismatches:", wrong_count)
    print(f"Failure Rate: {failure_rate:3f} %")
    if failure_rate > args.failure_threshold:
        print(f"[ERROR] {name} Computation Has Errors")
    else:
        print(f"[SUCCESS] {name} Computation Has Only Numerical Errors (Hopefully)")

def tiles_to_indices(sampled_pixels, likelihoods):
    no_tiles_in_row = math.ceil( H / TILE_SIZE)
    no_tiles_in_col = math.ceil( W / TILE_SIZE)
    indices = []
    likelihoods_array = []
    for tile in range(sampled_pixels.shape[0]):
        tile_row = tile // no_tiles_in_row
        tile_col = tile % no_tiles_in_col
        for sample_id, pixel in enumerate(sampled_pixels[tile]):
            pix_row = pixel // TILE_SIZE
            pix_col = pixel % TILE_SIZE
            global_row = tile_row * TILE_SIZE + pix_row
            global_col = tile_col * TILE_SIZE + pix_col
            if global_row < H and global_col < W:
                pix_id = global_row* W + global_col
                flattened_idx = pix_id * NUM_CHANNELS
                indices.append(flattened_idx)
                indices.append(flattened_idx + 1)
                indices.append(flattened_idx + 2)
                sample_likelihood = likelihoods[tile, sample_id]
                likelihoods_array.append(sample_likelihood)
                likelihoods_array.append(sample_likelihood)
                likelihoods_array.append(sample_likelihood)
    return indices, np.stack(likelihoods_array)

total_batches = args.batch_size
grad_gt = None
diag_gt = None
JTJv_gt = None
main_J_path = args.Jpath
main_J_cuda_path = args.J_cuda_path
H, W = 100, 100
TILE_SIZE = 16
NUM_CHANNELS = 3
all_sampled_pixels = np.load(os.path.join(main_J_cuda_path, "sampled_pixels.npy"))
all_likelihoods = np.load(os.path.join(main_J_cuda_path, "likelihoods.npy"))

SAMPLE_SIZE = all_sampled_pixels.shape[-1]

# GT Jacobian Vector Products
for batch in range(total_batches):
    sampled_pixels = all_sampled_pixels[batch]
    likelihoods = all_likelihoods[batch]

    indices, likelihoods = tiles_to_indices(sampled_pixels, likelihoods)
    path = os.path.join(main_J_path, f"residual_batch{batch}.npz")
    r = np.load(path)["arr_0"]
    path = os.path.join(main_J_path, f"Jacobian_batch{batch}.npz")
    J = np.load(path)["arr_0"]
    grad_gt = np.zeros((J.shape[1])) if grad_gt is None else grad_gt
    diag_gt = np.zeros((J.shape[1])) if diag_gt is None else diag_gt


    ## Sample Jacobian
    r = r[indices]
    J = J[indices]
    grad_gt += J.T @ (r / (likelihoods*SAMPLE_SIZE))
    scaled = J / (likelihoods.reshape(-1, 1)*SAMPLE_SIZE)
    diag_gt += np.einsum("ij,ji->i", J.T, scaled)

    del J

### Compare with CUDA Implementation

# Change the layout to CUDA version.
grad_gt_cuda_layout = grad_gt.reshape(-1, 14).T.flatten() #Do not lose the original layout, it will be needed
diag_gt = diag_gt.reshape(-1, 14).T.flatten()

### Check Gradient 
grad_path = os.path.join(args.J_cuda_path,"gradient_cuda.npy" )
gradient_cuda = np.load(grad_path)
is_gradient_close = np.allclose(gradient_cuda, grad_gt_cuda_layout, atol=args.atol, rtol=args.rtol)
print("[TEST]- Is Gradient Close: ", is_gradient_close)
if not is_gradient_close:
    info("Gradient", gradient_cuda, grad_gt_cuda_layout)

### Check Diagonal 
diag_path = os.path.join(args.J_cuda_path,"diag_cuda.npy" )
diag_cuda = np.load(diag_path)
is_diag_close = np.allclose(diag_cuda, diag_gt, atol=args.atol, rtol=args.rtol)
print("[TEST]- Is Diagonal Close: ", is_diag_close)
if not is_diag_close:
    info("Diagonal", diag_cuda, diag_gt)

## Check JTJv Results, where v is the residual, i.e. first step of CG
JTJv_path = os.path.join(args.J_cuda_path, "JTJv_cuda.npy" )
JTJv_cuda = np.load(JTJv_path)
for batch in range(total_batches):
    sampled_pixels = all_sampled_pixels[batch]
    likelihoods = all_likelihoods[batch]

    indices, likelihoods = tiles_to_indices(sampled_pixels, likelihoods)
    path = os.path.join(main_J_path, f"residual_batch{batch}.npz")
    r = np.load(path)["arr_0"]
    path = os.path.join(main_J_path, f"Jacobian_batch{batch}.npz")
    J = np.load(path)["arr_0"]
    JTJv_gt = np.zeros((J.shape[1])) if JTJv_gt is None else JTJv_gt

    ## Sample Jacobian
    r = r[indices]
    J = J[indices]
    Jv = J @ grad_gt
    JTJv_gt += J.T @ (Jv / (likelihoods*SAMPLE_SIZE))

    del J

# Change the layout to CUDA version.
JTJv_gt = JTJv_gt.reshape(-1, 14).T.flatten()

is_JTJ_close = np.allclose(JTJv_cuda, JTJv_gt, atol=args.atol, rtol=args.rtol_jtjv)
print("[TEST]- Is JTJv Close: ", is_JTJ_close)
if not is_JTJ_close:
    info("JTJv", JTJv_cuda, JTJv_gt)