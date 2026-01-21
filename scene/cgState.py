import torch
TILE_SIZE= 256
BLOCK_DIM = 16

class CGSolverState:
    def __init__(self, num, params) -> None:
        self.state = {}
        self.numberOfParams = num* 14 # There are 14 parameters for each Gaussian
        self.num_gaussians = num
        self.kernel = params.kernel
        self.batch_size = params.batch_size
        self.cg_iter = params.cg_iter
        self.loss_fn = params.loss_fn
        assert self.loss_fn in ["mse", "mse_approx_ssim"]

        self.state["d_colors"] = self.init((self.num_gaussians, 3))
        self.state["d_mean2D"] = self.init((self.num_gaussians, 3))
        self.state["d_cov2D"] = self.init((self.num_gaussians, 4))

        self.sampling_distribution = params.sampling_distribution
        self.sample_per_block = params.N_sample_per_tile
        self.tile_block_dim = params.tile_block_dimx, params.tile_block_dimy
        self.lambda_dssim = params.ssim_weight
        self.temperature = params.temperature
 

    def init(self, shp, dtype=torch.float32):
        return torch.zeros(shp, dtype=dtype, layout=torch.strided, device=torch.device("cuda"))
    def init_empty(self):
        return torch.tensor([])


    def set_scene_size(self, scene):
        self.width = scene.train_cameras[1.0][0].image_width
        self.height = scene.train_cameras[1.0][0].image_height


        self.width_blocks = (self.width + (self.tile_block_dim[0] -1)) // self.tile_block_dim[0]
        self.height_blocks = (self.height + (self.tile_block_dim[1] -1)) // self.tile_block_dim[1]

        self.total_blocks_sampled = self.width_blocks * self.height_blocks
        self.state["sampled_pixels"] = self.init((self.batch_size, self.total_blocks_sampled, self.sample_per_block ), torch.int32)
        self.state["likelihoods"] = self.init((self.batch_size, self.height_blocks * self.width_blocks, 16*16))
        self.state["n_of_gaussians_per_pixel"] = self.init((self.batch_size, self.height_blocks * self.width_blocks, 16*16), torch.int32)

        self.pixel_per_block = self.init((self.height_blocks * self.width_blocks))
        self.pixel_per_block[:] = TILE_SIZE
        
        self.full_image = torch.arange(256)
        self.full_image = self.full_image.repeat(self.total_blocks_sampled, 1).cuda()

        self.total_pixels = self.width * self.height * 3* self.batch_size

    def update(self, key: str, value):
        if key == "cgiter_sched" and self.cg_iter != value:
            print(f"[CG ITER]: Updated from {self.cg_iter} to {value}")
            self.cg_iter = value
            self.state["d_sumResidual"] = self.init((self.cg_iter, )) #Used to store d_r vs iter
        
        if key == "batchsize_sched" and self.batch_size != value:
            print(f"[Batch Size]: Updated from {self.batch_size} to {value}")
            self.batch_size = value

            self.state["sampled_pixels"] = self.init((self.batch_size, self.total_blocks_sampled, self.sample_per_block ), torch.int32)
            self.state["likelihoods"] = self.init((self.batch_size, self.height_blocks * self.width_blocks, 16*16))
            self.state["n_of_gaussians_per_pixel"] = self.init((self.batch_size, self.height_blocks * self.width_blocks, 16*16), torch.int32)
                
        if key == "samplesize_sched":
            if self.sample_per_block != value:
                print(f"[Sample Size]: Update from {self.sample_per_block} to {value}")
                self.sample_per_block = value
                self.state["sampled_pixels"] = self.init((self.batch_size, self.total_blocks_sampled, self.sample_per_block ), torch.int32)
            