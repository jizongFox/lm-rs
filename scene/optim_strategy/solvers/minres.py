import torch
from .base import BaseSolver
from diff_gaussian_rasterization import _RasterizeGaussians

class MinresSolver(BaseSolver):
    def __init__(self, size, solver="Minres", linear_iter=10, lambda_reg=1e-2):
        super().__init__(solver, linear_iter, lambda_reg)
        self.init_solver(size)
    
    def set_linear_iter(self, new_value):
        self.linear_iter = new_value
        
    def init_solver(self, size):
        self.x = torch.zeros((size, ), dtype=torch.float32, device="cuda")
        self.r1 = torch.zeros((size, ), dtype=torch.float32, device="cuda")

        self.diag = torch.zeros((size, ), dtype=torch.float32, device="cuda")
        self.gradient = torch.zeros((size, ), dtype=torch.float32, device="cuda")
        self.p = torch.zeros((size, ), dtype=torch.float32, device="cuda")
        self.Ap = torch.zeros((size, ), dtype=torch.float32, device="cuda")

        self.z = torch.zeros((size, ), dtype=torch.float32, device="cuda")
        self.inv_diag = torch.zeros((size, ), dtype=torch.float32, device="cuda")

    def clear_state(self):
        self.x[:] = 0
        self.r1[:] = 0
        self.diag[:] = 0
        self.gradient[:] = 0
        self.Ap[:] = 0

    def linear_solve(self, x0=None, debug=False):

        self.clear_state()
        # Set up y and v for the first Lanczos vector v1.
        # y  =  beta1 P' v1,  where  P = C**(-1).
        # v is really P' v1.
        _RasterizeGaussians.get_JTv_Diag(self.r1, self.diag)
        self.diag += self.lambda_reg # LM
        y = self.r1 / self.diag #Inversion

        beta1 = torch.dot(self.r1, y)
        beta1 = torch.sqrt(beta1)

        # Initialize other quantities
        oldb = 0
        beta = beta1
        r2 = self.r1.clone()
        n = self.r1.shape[0]
        dbar = 0
        epsln = 0
        phibar = beta1
        eps = 1e-10
        cs = -1
        sn = 0
        w = torch.zeros(n, dtype=torch.float32, device="cuda")
        w2 = torch.zeros(n, dtype=torch.float32, device="cuda")

        itn = 0
        while itn < self.linear_iter:
            itn += 1

            s = 1.0/beta
            v = s*y

            _RasterizeGaussians.get_JTJv(v, y)
            y = y + self.lambda_reg * v

            if itn >= 2:
                y = y - (beta/oldb)*self.r1

            alfa = torch.dot(v,y)
            y = y - (alfa/beta)*r2
            self.r1[:] = r2
            r2 = y
            y = r2 / self.diag ## Inverse
            oldb = beta
            beta = torch.dot(r2,y)
            beta = torch.sqrt(beta)

            # Apply previous rotation Qk-1 to get
            #   [deltak epslnk+1] = [cs  sn][dbark    0   ]
            #   [gbar k dbar k+1]   [sn -cs][alfak betak+1].

            oldeps = epsln
            delta = cs * dbar + sn * alfa   # delta1 = 0         deltak
            gbar = sn * dbar - cs * alfa   # gbar 1 = alfa1     gbar k
            epsln = sn * beta     # epsln2 = 0         epslnk+1
            dbar = - cs * beta   # dbar 2 = beta2     dbar k+1

            # Compute the next plane rotation Qk
            gamma = torch.sqrt(gbar**2 + beta**2)       # gammak
            gamma = max(gamma, eps)
            cs = gbar / gamma             # ck
            sn = beta / gamma             # sk
            phi = cs * phibar              # phik
            phibar = sn * phibar              # phibark+1

            # Update  x.

            denom = 1.0/gamma
            w1 = w2
            w2 = w
            w = (v - oldeps*w1 - delta*w2) * denom
            self.x = self.x + phi*w

        return self.x