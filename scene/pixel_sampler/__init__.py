from scene.pixel_sampler.loss_based_sampler import LossSampler
from scene.pixel_sampler.uniform_sampler import UniformSampler
from scene.pixel_sampler.gaussian_count_sampler import GaussianCountSampler





samplers =  {"uniform": UniformSampler, "loss_map": LossSampler, "gaussian_count": GaussianCountSampler}
