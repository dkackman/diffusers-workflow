import torch
import logging
import warnings
import diffusers
from packaging import version
from .log_setup import setup_logging
from .settings import resolve_path, load_settings

__version__ = "0.37.0"

settings = load_settings()


def get_device():
    """
    Detect and return the best available device for PyTorch operations.
    Priority: CUDA > MPS > CPU

    Returns:
        str: Device identifier ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def get_autocast_device_type():
    """
    Get the device type to use for torch.autocast.
    MPS doesn't support autocast, so use 'cpu' for MPS devices.

    Returns:
        str: Device type for autocast ('cuda' or 'cpu')
    """
    device = get_device()
    if device == "cuda":
        return "cuda"

    # MPS and CPU both use 'cpu' for autocast
    return "cpu"


def startup(log_level=None):
    device = get_device()

    # Suppress autocast warnings when using MPS (MPS doesn't support autocast,
    # and libraries may try to use it with 'cuda' device_type)
    if device == "mps":
        warnings.filterwarnings(
            "ignore",
            message=".*User provided device_type of 'cuda'.*",
            category=UserWarning,
            module="torch.amp.autocast_mode",
        )

    # Check if we have a GPU backend (CUDA or MPS)
    if device == "cpu":
        logging.warning(
            "No GPU backend available (CUDA or MPS). Running on CPU may be slow."
        )
    else:
        logging.info(f"Using device: {device}")

    if version.parse(torch.__version__) < version.parse("2.0.0"):
        raise Exception(
            f"Pytorch must be 2.0 or greater (found {torch.__version__}). Run install script. Quitting."
        )

    if log_level is not None:
        settings.log_level = log_level

    setup_logging(
        resolve_path(settings.log_filename),
        settings.log_level,
    )
    logging.info(f"Version {__version__}")
    logging.debug(f"Torch version {torch.__version__}")
    diffusers.logging.set_verbosity_error()

    torch.set_float32_matmul_precision("high")

    # CUDA-specific optimizations
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = False  # Prioritize performance
