import torch
from scene.pixel_sampler.base_sampler import BaseSampler
import matplotlib.pyplot as plt
import os

TILE_SIZE = 256
BLOCK_X = 16
BLOCK_Y = 16

class LossSampler(BaseSampler):
    def __init__(self, scene) -> None:
        super().__init__(scene.model_path)

    def sample(self, current_batch, cgState, **kwargs):
        sample_size = cgState.sample_per_block
        loss_map = kwargs.get("loss_map", None)
        loss_map = torch.sum(torch.abs(loss_map), dim=0) 
        h, w = loss_map.shape
        new_h = cgState.height_blocks * BLOCK_Y
        new_w = cgState.width_blocks * BLOCK_Y
        if (new_h != h or new_w != w):
            loss_map = torch.nn.functional.pad(loss_map, (0, new_w-w, 0, new_h-h ), mode='constant', value=0)
        loss_map = loss_map + 1e-8 # Safeguard against 0 loss
        visualize = kwargs.get("visualize", False)

        if visualize: 
            self.visualize_loss_map(loss_map)

        loss_map = loss_map.view(cgState.height_blocks, BLOCK_Y, cgState.width_blocks, BLOCK_X).permute(0, 2, 1, 3).reshape(-1, TILE_SIZE)
        probabilities = torch.softmax(loss_map / cgState.temperature, dim=-1)
        sampled_pixels = torch.multinomial(probabilities, sample_size, replacement=True)
        cgState.state["sampled_pixels"][current_batch] = sampled_pixels.int()
        cgState.state["likelihoods"][current_batch] = probabilities

        if visualize:
            probabilities_viz = (
                            probabilities.reshape(cgState.height_blocks, cgState.width_blocks, BLOCK_Y, BLOCK_X)
                            .permute(0, 2, 1, 3).reshape(cgState.height_blocks * BLOCK_Y, cgState.width_blocks*BLOCK_X)
                            ).cpu().numpy()
            self.visualize_prob(probabilities_viz)

        
    
    def visualize_loss_map(self, loss_map):
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(loss_map.cpu().numpy(), cmap='viridis')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Loss Value')

        # Set axis labels and title
        ax.set_title('Loss Map')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')
        fig.tight_layout()
    
        plt.savefig(os.path.join(self.save_dir, f"img_{self.get_iteration()}.png"), dpi=300)
        plt.close(fig)