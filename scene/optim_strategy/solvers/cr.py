import torch
from .base import BaseSolver
from diff_gaussian_rasterization import _RasterizeGaussians


# Preconditioned Conjugate Resdiauls Solver
class CRSolver(BaseSolver):
    def __init__(self, size, solver="CR", linear_iter=10, lambda_reg=1e-2):
        super().__init__(solver, linear_iter, lambda_reg)
        self.init_solver(size)

    def set_linear_iter(self, new_value):
        self.linear_iter = new_value

    def init_solver(self, size):
        self.x = torch.zeros((size,), dtype=torch.float32, device="cuda")
        self.r = torch.zeros((size,), dtype=torch.float32, device="cuda")
        self.diag = torch.zeros((size,), dtype=torch.float32, device="cuda")
        self.gradient = torch.zeros((size,), dtype=torch.float32, device="cuda")
        self.p = torch.zeros((size,), dtype=torch.float32, device="cuda")
        self.Ap = torch.zeros((size,), dtype=torch.float32, device="cuda")
        self.Ar = torch.zeros((size,), dtype=torch.float32, device="cuda")
        self.inv_diag = torch.zeros((size,), dtype=torch.float32, device="cuda")

    def clear_state(self):
        self.x[:] = 0
        self.r[:] = 0
        self.diag[:] = 0
        self.gradient[:] = 0
        self.Ar[:] = 0

    def linear_solve(self, x0=None, debug=False):
        with torch.no_grad():
            self.clear_state()

            if debug:
                _RasterizeGaussians.get_JTv_Diag(self.r, self.diag)
                _RasterizeGaussians.get_JTJv(self.r, self.Ap)
                return None

            _RasterizeGaussians.get_JTv_Diag(self.r, self.diag)
            self.diag += self.lambda_reg  # LM

            torch.div(1.0, self.diag, out=self.inv_diag)
            torch.mul(self.inv_diag, self.r, out=self.r)
            self.p[:] = self.r

            _RasterizeGaussians.get_JTJv(self.r, self.Ar)
            self.Ar += self.r * self.lambda_reg
            self.Ap[:] = self.Ar
            r_dot_Ar_old = torch.dot(self.r, self.Ar)

            for i in range(self.linear_iter):
                alpha_denom = torch.dot(self.Ap, self.Ap * self.inv_diag)
                alpha = r_dot_Ar_old / alpha_denom
                self.x = self.x + alpha * self.p

                if i == (self.linear_iter - 1):
                    return self.x  # Dont do extra matvec

                self.r = self.r - alpha * self.inv_diag * self.Ap
                _RasterizeGaussians.get_JTJv(self.r, self.Ar)
                self.Ar += self.r * self.lambda_reg

                r_dot_Ar_new = torch.dot(self.r, self.Ar)
                beta = r_dot_Ar_new / r_dot_Ar_old
                self.p = self.r + beta * self.p
                self.Ap = self.Ar + beta * self.Ap
                r_dot_Ar_old = r_dot_Ar_new.clone()
