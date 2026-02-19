class BaseSolver:
    def __init__(self, solver="CG", linear_iter=10, lambda_reg=1e-2):
        self.solver = solver
        self.linear_iter = linear_iter
        self.lambda_reg = lambda_reg

    def init_solver(self, size):
        pass
