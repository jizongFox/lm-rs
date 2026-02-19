import torch
from scene.pixel_sampler.base_sampler import BaseSampler

TILE_SIZE = 256
BLOCK_X = 16
BLOCK_Y = 16


class UniformSampler(BaseSampler):
    def __init__(self, scene) -> None:
        super().__init__(scene.model_path)

    def sample(self, current_batch, cgState, **kwargs):

        sample_size = cgState.sample_per_block
        total_tiles = cgState.total_blocks_sampled
        max_pixels = cgState.tile_block_dim[0] * cgState.tile_block_dim[1]
        sampled_pixels = torch.randint(
            low=0, high=max_pixels, size=(total_tiles, sample_size), device="cuda"
        )
        cgState.state["sampled_pixels"][current_batch] = sampled_pixels
        cgState.state["likelihoods"][current_batch] = 1 / 256

    def visualize():
        pass
