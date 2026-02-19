from scene.camera_sampler.random_sampler import RandomSampler
from scene.camera_sampler.cluster_sampler import ClusterSampler
from scene.camera_sampler.sequential_sampler import SequentialSampler
from scene.camera_sampler.clusterdir_sampler import ClusterDirSampler

samplers = {
    "random": RandomSampler,
    "cluster": ClusterSampler,
    "sequential": SequentialSampler,
    "clusterdir": ClusterDirSampler,
}
