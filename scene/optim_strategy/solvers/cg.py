import torch
from .base import BaseSolver
from diff_gaussian_rasterization import _RasterizeGaussians


# Preconditioned Conjugate Gradient Solver
class CGSolver(BaseSolver):
    def __init__(
        self,
        size,
        solver="CG",
        linear_iter=1,
        lambda_reg=1e-2,
        levenberg_type="identity",
    ):
        super().__init__(solver, linear_iter, lambda_reg)
        self.init_solver(size)
        self.size = (
            size // 14
        )  # There are 14 params, 3 location, 3 scale, 4 rot, 3 color, 1 opac
        assert (
            levenberg_type == "identity" or levenberg_type == "diagonal"
        ), "Levenberg Regularizer is not supported"
        self.levenberg_type = levenberg_type
        print("Lambda is set to:", self.lambda_reg)
        self.iter = 0

    def set_linear_iter(self, new_value):
        self.linear_iter = new_value

    def init_solver(self, size):
        self.x = torch.zeros((size,), dtype=torch.float32, device="cuda")
        self.r = torch.zeros((size,), dtype=torch.float32, device="cuda")
        self.diag = torch.zeros((size,), dtype=torch.float32, device="cuda")
        self.gradient = torch.zeros((size,), dtype=torch.float32, device="cuda")
        self.p = torch.zeros((size,), dtype=torch.float32, device="cuda")
        self.Ap = torch.zeros((size,), dtype=torch.float32, device="cuda")

        self.z = torch.zeros((size,), dtype=torch.float32, device="cuda")
        self.inv_diag = torch.zeros((size,), dtype=torch.float32, device="cuda")

    def clear_state(self):
        self.x[:] = 0
        self.r[:] = 0
        self.diag[:] = 0
        self.gradient[:] = 0
        self.Ap[:] = 0

    def linear_solve(self, debug=False):
        with torch.no_grad():
            self.clear_state()

            if debug:
                _RasterizeGaussians.get_JTv(self.r, self.size)
                _RasterizeGaussians.get_Diag(self.diag, self.size)
                _RasterizeGaussians.get_JTJv(self.r, self.Ap, self.size)
                return None

            _RasterizeGaussians.get_JTv(self.r, self.size)
            _RasterizeGaussians.get_Diag(self.diag, self.size)
            self.gradient[:] = self.r
            if self.levenberg_type == "identity":
                self.diag += self.lambda_reg  # LM
            elif self.levenberg_type == "diagonal":
                self.diag += self.lambda_reg * self.diag
                self.diag += 1e-6  # Avoid zero diagonal when inverting in next step

            torch.div(1.0, self.diag, out=self.inv_diag)
            torch.mul(self.inv_diag, self.r, out=self.z)
            self.p[:] = self.z
            r_dot_z_old = torch.dot(self.r, self.z)

            for i in range(self.linear_iter):
                _RasterizeGaussians.get_JTJv(self.p, self.Ap, self.size)
                if self.levenberg_type == "identity":
                    self.Ap += self.p * self.lambda_reg  # LM
                elif self.levenberg_type == "diagonal":
                    self.Ap += self.p * self.lambda_reg * self.diag

                alpha = r_dot_z_old / torch.dot(self.p, self.Ap)
                torch.add(self.x, self.p, alpha=alpha, out=self.x)
                torch.add(self.r, self.Ap, alpha=-alpha, out=self.r)

                torch.mul(self.inv_diag, self.r, out=self.z)
                r_dot_z_new = torch.dot(self.r, self.z)
                beta = r_dot_z_new / r_dot_z_old
                self.p.mul_(beta).add_(self.z)
                r_dot_z_old = r_dot_z_new.clone()

        return self.x
