import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


class BaseSampler:
    def __init__(self, path) -> None:
        self.save_dir = os.path.join(path, "likelihoods")
        os.makedirs(self.save_dir, exist_ok=True)

        self.iteration = 1

    def iter(self):
        self.iteration = self.iteration + 1

    def get_iteration(self):
        return self.iteration

    def visualize_prob(self, probabilities_numpy):

        fig, ax = plt.subplots(figsize=(6, 5))
        threshold = np.percentile(probabilities_numpy, 95)
        im2 = ax.imshow(
            probabilities_numpy, norm=Normalize(vmin=0, vmax=threshold), cmap="hot"
        )
        cbar2 = fig.colorbar(im2, ax=ax)
        cbar2.set_label("Likelihood")

        ax.set_title("Likelihood Map")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")

        fig.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"prob_{self.iteration}.png"), dpi=300)
        plt.close(fig)

        self.iter()
