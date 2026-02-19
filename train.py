#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

#
import sys
import os

import torch
from utils.loss_utils import ssim, l2_loss
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from argparse import ArgumentParser, Namespace
from arguments import (
    ModelParams,
    PipelineParams,
    OptimizationParams,
    GaussNewtonOptimizationParams,
    SchedulerParameters,
)
import numpy as np
from scene.gaussianOptimizerHandler import OptimizerHandler
from scene.step.gauss_newton_step import gauss_newton_step
from scene.step.adam_step import adam_step
from scene.step.test_step import save_jacob
import scene.camera_sampler as CameraSampler
import scene.pixel_sampler as PixelSampler

from utils.image_utils import psnr
from lpipsPyTorch import lpips_v2

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import time


def prepare_output_and_logger(args, gn_args=None):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(os.path.join(args.model_path, "cg_curves"), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, "metrics"), exist_ok=True)

    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    if gn_args != None:
        with open(os.path.join(args.model_path, "gn_args"), "w") as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(gn_args))))

    os.makedirs(os.path.join(args.model_path, "cg_curves"), exist_ok=True)
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def densify_prune_adam(
    opt,
    gaussians: GaussianModel,
    scene,
    iteration,
    visibility_filter,
    radii,
    viewspace_point_tensor,
    optimization_method,
    visualize: bool = False,
):
    if iteration < opt.densify_until_iter:
        # Keep track of max radii in image-space for pruning
        gaussians.max_radii2D[visibility_filter] = torch.max(
            gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
        )

        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if (
            iteration > opt.densify_from_iter
            and iteration % opt.densification_interval == 0
        ):
            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            gaussians.densify_and_prune(
                opt.densify_grad_threshold,
                0.005,
                scene.cameras_extent,
                size_threshold,
                optimization_method,
            )
            print(f"Iter {iteration} Number of Gaussians: ", gaussians._xyz.shape[0])
            print("Memory: ", torch.cuda.max_memory_allocated() / 1e9, " GB")

        if (
            iteration % (opt.opacity_reset_interval) == 0
        ):  # or (dataset.white_background and iteration == opt.densify_from_iter) I dont use white
            gaussians.reset_opacity(optimization_method)


def save_tensors(
    gaussians: GaussianModel,
    params: GaussNewtonOptimizationParams,
    optimizerHandler,
    path: str,
):
    out_dir = os.path.join(path, "cuda_results")
    os.makedirs(out_dir, exist_ok=True)
    np.save(
        os.path.join(out_dir, "gradient_cuda.npy"),
        optimizerHandler.solver.r.cpu().numpy(),
    )
    np.save(
        os.path.join(out_dir, "diag_cuda.npy"),
        optimizerHandler.solver.diag.cpu().numpy(),
    )
    np.save(
        os.path.join(out_dir, "JTJv_cuda.npy"), optimizerHandler.solver.Ap.cpu().numpy()
    )
    np.save(
        os.path.join(out_dir, "likelihoods.npy"),
        gaussians.cgState.state["likelihoods"].cpu().numpy(),
    )
    np.save(
        os.path.join(out_dir, "sampled_pixels.npy"),
        gaussians.cgState.state["sampled_pixels"].cpu().numpy(),
    )

    print("CUDA Results are Saved Succesfully")
    exit(0)


def training(
    dataset,
    opt,
    pipe,
    log_dict,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    optimization_method: str,
    gn_opt_params: GaussNewtonOptimizationParams,
    enable_densify: bool,
    save: bool,
    sched_params=None,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, gn_opt_params)
    gaussians = GaussianModel(dataset.sh_degree, optimization_method)
    scene = Scene(dataset, gaussians, opt)
    gaussians.training_setup(opt, gn_opt_params)

    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt, optimization_method, gn_opt_params)
        print("Ckpt Loaded. Number of Gaussians is, ", gaussians._xyz.shape[0])

    if optimization_method == "cg-gpu":
        gaussians.cgState.set_scene_size(scene)
        gaussians.initialize_schedulers(sched_params)
    optimization_method = (
        "first_order"
        if optimization_method in ["adam", "rmsprop", "sgd"]
        else optimization_method
    )
    optimizerHandler = OptimizerHandler(
        gaussians,
        optimization_method,
        gn_opt_params,
        opt,
        scene.model_path,
        scene.cameras_extent,
    ).getOptimizer()
    # optimizerHandler_adam = OptimizerHandler(gaussians, "adam", gn_opt_params, opt, scene.model_path).getOptimizer()

    camera_sampler_dict = {
        "batch_size": gn_opt_params.batch_size,
        "path": scene.model_path,
    }
    camera_sampler = CameraSampler.samplers[gn_opt_params.camera_sampler](
        scene.getTrainCameras(), camera_sampler_dict
    )

    pixel_sampler = PixelSampler.samplers[gn_opt_params.sampling_distribution](scene)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    first_iter += 1
    viewpoint_stack = None
    total_elapsed = 0
    total_steps = 0

    with torch.no_grad():
        training_report(
            tb_writer,
            1,
            0,
            log_dict,
            scene,
            render,
            (pipe, background),
            optimization_method,
            gaussians,
            dataset.model_path,
        )

    for iteration in range(first_iter, opt.iterations + 1):
        total_steps = total_steps + 1
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    with torch.no_grad():
                        net_image = render(
                            custom_cam,
                            gaussians,
                            pipe,
                            background,
                            scaling_modifier=scaling_modifer,
                        )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                    (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception:
                network_gui.conn = None
        if optimization_method == "first_order":
            batch_size = gn_opt_params.batch_size
            new_lr = gaussians.update_learning_rate(iteration)
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()
            loss, elapsed, viewspace_point_tensor, visibility_filter, radii = adam_step(
                camera_sampler,
                iteration,
                debug_from,
                opt,
                scene,
                viewpoint_stack,
                gaussians,
                pipe,
                background,
                optimizerHandler,
                batch_size,
            )

        elif optimization_method == "cg-gpu":
            if gaussians.cgState.kernel == -1:
                save_jacob(
                    gn_opt_params.batch_size,
                    scene,
                    viewpoint_stack,
                    gaussians,
                    pipe,
                    background,
                    camera_sampler,
                    dataset.model_path,
                )
            else:
                if not gn_opt_params.disable_scheds:
                    gaussians.scheduler_step(
                        iteration, camera_sampler, optimizerHandler
                    )
                batch_size = gaussians.cgState.batch_size
                loss, elapsed = gauss_newton_step(
                    pixel_sampler,
                    camera_sampler,
                    iteration,
                    debug_from,
                    opt,
                    gaussians,
                    pipe,
                    background,
                    optimizerHandler,
                    gn_opt_params,
                )

        if save:
            save_tensors(gaussians, gn_opt_params, optimizerHandler, dataset.model_path)

        with torch.no_grad():
            total_elapsed = total_elapsed + elapsed
            training_report(
                tb_writer,
                iteration,
                total_elapsed,
                log_dict,
                scene,
                render,
                (pipe, background),
                optimization_method,
                gaussians,
                dataset.model_path,
            )

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if enable_densify:  # Always disabled in this work
                if optimization_method == "first_order":
                    densify_prune_adam(
                        opt,
                        gaussians,
                        scene,
                        iteration,
                        visibility_filter,
                        radii,
                        viewspace_point_tensor,
                        optimization_method,
                    )

            start_optim = time.time()
            if iteration < opt.iterations:
                optimizerHandler.step(gaussians)
            torch.cuda.synchronize()
            elapsed = time.time() - start_optim
            total_elapsed = total_elapsed + elapsed
            if optimization_method == "cg-gpu" and iteration in log_dict["metrics"]:
                optimizerHandler.plot_lr()

            if iteration in checkpoint_iterations:
                torch.save(
                    (gaussians.capture(optimization_method), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )

    print("Per step training time: ", (total_elapsed * 1000 / (total_steps)), " ms")
    print("Total training time (forward + backward + step): ", total_elapsed, " sec")
    print("Memory Allocated", torch.cuda.memory_allocated() * 1e-9)

    if tb_writer:
        tb_writer.close()


time_list = {"test": [], "train": []}
ssim_list = {"test": [], "train": []}
l2_list = {"test": [], "train": []}
psnr_list = {"test": [], "train": []}
lpips_list = {"test": [], "train": []}
step_list = {"test": [], "train": []}


def training_report(
    tb_writer,
    iteration,
    elapsed,
    log_dict,
    scene: Scene,
    renderFunc,
    renderArgs,
    optimization_method,
    gaussians,
    path,
):
    # Report test and samples of training set
    testing_iterations = log_dict["metrics"]
    log_image_iterations = log_dict["images"]
    metric_path = os.path.join(path, "metrics")

    log_images = iteration in log_image_iterations

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        train_cameras = scene.getTrainCameras()
        total_cameras = len(train_cameras)
        num_selected_cameras = 20
        step = total_cameras // num_selected_cameras
        selected_cameras = [
            train_cameras[i * step] for i in range(num_selected_cameras)
        ]

        # Define validation configurations with selected cameras
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {"name": "train", "cameras": selected_cameras},
        )
        l2_test = None
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l2_test = 0.0
                ssim_test = 0.0
                psnr_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"],
                        0.0,
                        1.0,
                    )
                    if log_images and tb_writer:
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            (image)[None],
                            global_step=iteration,
                        )

                    l2_test += l2_loss(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    lpips_test += lpips_v2(image, gt_image).mean().double()

                ssim_test /= len(config["cameras"])
                l2_test /= len(config["cameras"])
                psnr_test /= len(config["cameras"])
                lpips_test /= len(config["cameras"])

                print(
                    "\n[ITER {}] Evaluating {}: L2 {} SSIM {} PSNR {} LPIPS {}".format(
                        iteration,
                        config["name"],
                        l2_test,
                        ssim_test,
                        psnr_test,
                        lpips_test,
                    )
                )
                print(f"Number of Gaussians is : {gaussians._xyz.shape[0]}")
                print("Forward + Backward Time ", elapsed, flush=True)
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/optim_steps - ssim",
                        ssim_test,
                        iteration,
                        walltime=elapsed,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/optim_steps - l2",
                        l2_test,
                        iteration,
                        walltime=elapsed,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/optim_steps - psnr",
                        psnr_test,
                        iteration,
                        walltime=elapsed,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/optim_steps - lpips",
                        lpips_test,
                        iteration,
                        walltime=elapsed,
                    )

                ssim_list[config["name"]].append(ssim_test.item())
                time_list[config["name"]].append(elapsed)
                l2_list[config["name"]].append(l2_test.item())
                psnr_list[config["name"]].append(psnr_test.item())
                lpips_list[config["name"]].append(lpips_test.item())
                step_list[config["name"]].append(iteration)

                np.save(
                    os.path.join(metric_path, f"{config['name']}_ssim.npy"),
                    np.asarray(ssim_list[config["name"]], dtype=np.float32),
                )
                np.save(
                    os.path.join(metric_path, f"{config['name']}_time.npy"),
                    np.asarray(time_list[config["name"]], dtype=np.float32),
                )
                np.save(
                    os.path.join(metric_path, f"{config['name']}_mse.npy"),
                    np.asarray(l2_list[config["name"]], dtype=np.float32),
                )
                np.save(
                    os.path.join(metric_path, f"{config['name']}_psnr.npy"),
                    np.asarray(psnr_list[config["name"]], dtype=np.float32),
                )
                np.save(
                    os.path.join(metric_path, f"{config['name']}_lpips.npy"),
                    np.asarray(lpips_list[config["name"]], dtype=np.float32),
                )
                np.save(
                    os.path.join(metric_path, f"{config['name']}_iter.npy"),
                    np.asarray(step_list[config["name"]], dtype=np.int32),
                )

        if tb_writer:
            tb_writer.add_scalar(
                "total_points", scene.gaussians.get_xyz.shape[0], iteration
            )
            if log_images:
                tb_writer.add_histogram(
                    "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
                )
                tb_writer.add_histogram(
                    "scene/color_histogram",
                    scene.gaussians._features_dc.flatten(),
                    iteration,
                )
                tb_writer.add_histogram(
                    "scene/xyz_histogram", scene.gaussians._xyz.flatten(), iteration
                )
                tb_writer.add_histogram(
                    "scene/scale_histogram",
                    scene.gaussians._scaling.flatten(),
                    iteration,
                )

        # tb_writer.add_scalar('iter_time', elapsed, iteration)
        print("Max Memory Allocated:  ", torch.cuda.max_memory_allocated() * 1e-9)
        torch.cuda.empty_cache()


def save_params_to_file(args, filename="params.py"):
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, filename), "w") as f:
        f.write("# Parameters for experiment\n")
        for key, value in vars(args).items():
            # Format the value to match Python literal representation
            if isinstance(value, str):
                value = f"'{value}'"  # Add quotes around strings
            elif isinstance(value, list):
                value = f"{value}"  # Lists will be represented as Python lists

            f.write(f"{key} = {value}\n")


def process_sched_params(sched_opts, gn_op):
    processed_params = {}
    for key, value in vars(sched_opts).items():
        split = value.split(",")
        if len(split[0]) == 0:
            processed_params[key] = []
        else:
            try:
                processed_params[key] = [int(item.strip()) for item in split]
            except:
                processed_params[key] = [float(item.strip()) for item in split]

    return processed_params


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    gn_op = GaussNewtonOptimizationParams(parser)
    sched_opts = SchedulerParameters(parser)
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=48766)
    parser.add_argument("--enable_viewer", action="store_true")

    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=True)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[200])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--optimization_method", default="cg-gpu", type=str)
    parser.add_argument("--enable_densify", action="store_true", default=False)
    parser.add_argument("--log_freq", type=int, default=1000)
    parser.add_argument("--log_image_freq", type=int, default=1000)
    parser.add_argument("--save", type=bool, default=False)

    args = parser.parse_args(sys.argv[1:])

    if args.log_freq > 0:
        log_iterations = list(range(0, args.iterations, args.log_freq))
        log_iterations[0] = 1
        log_iterations.append(args.iterations)
    else:
        log_iterations = [-1]
    log_images_iterations = list(range(0, args.iterations, args.log_image_freq))
    log_images_iterations.append(args.iterations)

    log_dict = {"metrics": log_iterations, "images": log_images_iterations}
    args.save_iterations.append(args.iterations)

    if args.optimization_method == "cg-gpu":
        sched_opts = process_sched_params(sched_opts.extract(args), gn_op)

    save_params_to_file(args)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    if args.enable_viewer:
        network_gui.init(args.ip, args.port)
    start_train = time.time()
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        log_dict,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.optimization_method,
        gn_op.extract(args),
        args.enable_densify,
        args.save,
        sched_opts,
    )

    end_train = time.time()
    print("Time passed (including io and eval) ", end_train - start_train, " second")
