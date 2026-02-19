import torch
from functools import wraps


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


@torch.no_grad()
def timing(f):
    timer_start = torch.cuda.Event(enable_timing=True)
    timer_end = torch.cuda.Event(enable_timing=True)

    @wraps(f)
    def wrap(*args, **kw):
        timer_start.record()
        result = f(*args, **kw)
        timer_end.record()
        torch.cuda.synchronize()
        print(
            "function:%r took: %2.6f msec"
            % (f.__name__, timer_start.elapsed_time(timer_end)),
            flush=True,
        )
        print(f"Memory usage: {torch.cuda.memory_allocated() * 1e-9} GB", flush=True)
        return result

    return wrap


type_to_bytes = {torch.long: 8, torch.int32: 4, torch.float: 4, torch.float32: 4}


@torch.no_grad()
def get_size(aTensor):
    total_nbytes = torch.numel(aTensor) * type_to_bytes[aTensor.dtype]
    gb = total_nbytes / 1e9
    print(f"Tensor takes {gb} GB space")
