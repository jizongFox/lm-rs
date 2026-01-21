from .cg import CGSolver
from .cr import CRSolver
from .minres import MinresSolver

all_solvers = {"CG": CGSolver, "CR": CRSolver, "MINRES": MinresSolver}

def get_solver(name):
    return all_solvers[name.upper()]