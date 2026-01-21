import numpy as np
import matplotlib.pyplot as plt


class BasePieceWiseScheduler:
    def __init__(self, setup_dict):
        """
        Initializes the PieceWise Scheduler.
        
        Parameters:
        - breakpoints: List of step thresholds where CG iterations change.
        - values: List of CG iterations corresponding to each breakpoint range.
        """
        breakpoints = setup_dict["breakpoints"]
        values = setup_dict["values"]

        assert len(breakpoints) + 1 == len(values), "Lenght of breakpoints should be 1 more than lenght of values"

        self.breakpoints = np.array(breakpoints)
        self.values = np.array(values)

    def get_value(self, step):
        return self.values[np.searchsorted(self.breakpoints, step, side="right")]

    def plot_schedule(self, total_steps):
        """Plots the piecewise constant CG iteration schedule."""
        steps = np.arange(0, total_steps + 1)
        cg_values = np.array([self.get_cg_iterations(s) for s in steps])

        plt.plot(steps, cg_values, drawstyle="steps-post", label="CG Iterations")
        plt.xlabel("NonLinear Iterations")
        plt.ylabel("CG Iterations")
        plt.title("Dynamic Piecewise Constant Scheduling for CG Iterations")
        plt.legend()
        plt.show()

class CGIterScheduler(BasePieceWiseScheduler):
    def __init__(self, setup_dict):
        super().__init__(setup_dict)

class BatchSizeScheduler(BasePieceWiseScheduler):
    def __init__(self, setup_dict):
        super().__init__(setup_dict)

class SampleSizeScheduler(BasePieceWiseScheduler):
    def __init__(self, setup_dict):
        super().__init__(setup_dict)

class LambdaScheduler(BasePieceWiseScheduler):
    def __init__(self, setup_dict):
        super().__init__(setup_dict)
