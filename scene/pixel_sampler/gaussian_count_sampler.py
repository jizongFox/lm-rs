import torch
import os
import matplotlib.pyplot as plt
from scene.pixel_sampler.base_sampler import BaseSampler

TILE_SIZE = 256
BLOCK_X = 16
BLOCK_Y = 16

class GaussianCountSampler(BaseSampler):
    def __init__(self, scene) -> None:
        super().__init__(scene.model_path)

    def sample(self, current_batch, cgState, **kwargs):
        
        sample_size = cgState.sample_per_block
        n_gaussians = cgState.state["n_of_gaussians_per_pixel"][current_batch]
        n_gaussians = n_gaussians + 1
        probabilities = n_gaussians / n_gaussians.sum(dim=-1, keepdim=True)
        sampled_pixels = torch.multinomial(probabilities, sample_size, replacement=True)
        cgState.state["sampled_pixels"][current_batch] = sampled_pixels.int()
        cgState.state["likelihoods"][current_batch] = probabilities
        visualize = kwargs.get("visualize", False)
        if visualize:
            self.visualize_gaussian_map(n_gaussians, probabilities, cgState)

            
    
    def visualize_gaussian_map(self, n_gaussians, probabilities, cgState):
        n_gaussians_numpy = (
        n_gaussians.reshape(cgState.height_blocks, cgState.width_blocks, BLOCK_Y, BLOCK_X)
        .permute(0, 2, 1, 3).reshape(cgState.height_blocks * BLOCK_Y, cgState.width_blocks*BLOCK_X)
        ).cpu().numpy()
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(n_gaussians_numpy, cmap='viridis')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Loss Value')

        # Set axis labels and title
        ax.set_title('Gaussian Count Map')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        fig.tight_layout()

        plt.savefig(os.path.join(self.save_dir, f"img_{self.get_iteration()}.png"), dpi=300)
        plt.close(fig)
        probabilities_viz = (
            probabilities.reshape(cgState.height_blocks, cgState.width_blocks, BLOCK_Y, BLOCK_X)
            .permute(0, 2, 1, 3).reshape(cgState.height_blocks * BLOCK_Y, cgState.width_blocks*BLOCK_X)
            ).cpu().numpy()
        
        self.visualize_prob(probabilities_viz)
