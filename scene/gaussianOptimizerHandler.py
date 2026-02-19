from scene.optim_strategy.cgOptimizer import CGOptimizer
from scene.optim_strategy.adamOptimizer import AdamOptimizer


class OptimizerHandler:
    def __init__(
        self,
        gaussians,
        optimization_method: str,
        params=None,
        adam_params=None,
        path=None,
        cameras_extend=None,
    ) -> None:
        if optimization_method == "first_order":
            self.optimizer = AdamOptimizer(adam_params, params.loss_fn)
        elif optimization_method == "cg-gpu":
            self.optimizer = CGOptimizer(gaussians, params, path, cameras_extend)
        elif optimization_method == "lbfgs":
            pass
        else:
            raise "Not Implemented"

    def getOptimizer(self):
        return self.optimizer

    def get_shift(self, image, gt_image, gaussians):
        return self.optimizer.get_shift(image, gt_image, gaussians)

    def step(self, gaussians):
        self.optimizer.step(gaussians)
