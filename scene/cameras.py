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
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor, nn

from utils.graphics_utils import (
    getProjectionMatrix,
    getProjectionMatrixShift,
    getWorld2View2,
)


class Camera(nn.Module):
    def __init__(
        self,
        *,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        gt_alpha_mask,
        image_name,
        image_path: str,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
        cx: float,
        cy: float,
        fx: float,
        fy: float,
        width: int,
        height: int,
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy
        self.image_width = width
        self.image_height = height

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")

        # Use self.image (lazy loading property) for original_image
        self.original_image = self.image

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        )
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        )

        self.projection_matrix = (
            getProjectionMatrixShift(
                znear=self.znear,
                zfar=self.zfar,
                focal_x=self.fx,
                focal_y=self.fy,
                cx=self.cx,
                cy=self.cy,
                width=self.image_width,
                height=self.image_height,
                device=self.data_device,
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @property
    def world2cam(self):
        return self.world_view_transform.transpose(0, 1)

    @property
    def cam2world(self):
        return self.world2cam.inverse()

    @property
    def intrinsic_matrix(self):
        return torch.tensor(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
            dtype=torch.float32,
            device=self.data_device,
        )

    @property
    def camera_center2(self):
        return self.cam2world[:3, 3]

    def extra_repr(self) -> str:
        return f"Camera {self.uid}: {self.image_name} ({self.image_width}x{self.image_height}) FoVx: {self.FoVx}, FoVy: {self.FoVy}, znear: {self.znear}, zfar: {self.zfar})"

    @property
    def image(self) -> Float[Tensor, "3 h w"]:
        assert Path(self.image_path).exists(), f"Image {self.image_path} not found"
        image_pil = Image.open(self.image_path).convert("RGB")
        image_pil = image_pil.resize(
            (self.image_width, self.image_height), Image.Resampling.LANCZOS
        )
        image_np = np.array(image_pil).transpose(2, 0, 1)
        return torch.from_numpy(image_np).float() / 255.0


class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
