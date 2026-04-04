# Based on detectron2.utils.memory

import logging
from contextlib import contextmanager
from functools import wraps
import torch


@contextmanager
def _ignore_torch_cuda_oom():
    """
    A context which ignores CUDA OOM exception from pytorch.
    """
    try:
        yield
    except torch.OutOfMemoryError as e:
        pass

def retry_if_cuda_oom(func, device=None):
    """
    Makes a function retry itself after encountering
    pytorch's CUDA OOM error.
    It will first retry after calling `torch.cuda.empty_cache()`.

    If that still fails, it will then retry by trying to convert inputs to CPUs.
    In this case, it expects the function to dispatch to CPU implementation.

    The return values may become CPU tensors as well. If device is specified, will attempt to transfer them to the device.
    Otherwise, it is up to the caller to handle any potential device mismatches.

    Args:
        func: a stateless callable that takes tensor-like objects as arguments
        device: torch.device to copy return output to, assuming return object has attributes `device` and `to`.
    Returns:
        a callable which retries `func` if OOM is encountered.

    Examples:
    ::
        output = retry_if_cuda_oom(some_torch_function, device='cuda')(input1, input2)

    Note:
        1. When converting inputs to CPU, it will only look at each argument and check
           if it has `.device` and `.to` for conversion. Nested structures of tensors
           are not supported.

        2. Since the function might be called more than once, it has to be
           stateless.
    """

    def maybe_to_cpu(x):
        try:
            like_gpu_tensor = x.device.type == "cuda" and hasattr(x, "to")
        except AttributeError:
            like_gpu_tensor = False
        if like_gpu_tensor:
            return x.to(device="cpu")
        else:
            return x

    @wraps(func)
    def wrapped(*args, **kwargs):
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Clear cache and retry
        torch.cuda.empty_cache() # noqa
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Try on CPU. This slows down the code significantly, therefore print a notice.
        logger = logging.getLogger(__name__) # noqa
        logger.info("Attempting to copy inputs of {} to CPU due to CUDA OOM".format(str(func)))
        new_args = (maybe_to_cpu(x) for x in args)
        new_kwargs = {k: maybe_to_cpu(v) for k, v in kwargs.items()}
        retval = func(*new_args, **new_kwargs)
        if device is not None:
            if hasattr(retval, "device") and hasattr(retval, "to"):
                with _ignore_torch_cuda_oom():
                    return retval.to(device)
                logger.info(f"Failed to copy output to device {device}; returning on {retval.device}") # noqa
        return retval

    return wrapped
