from .settings import resolve_path, load_settings
from .log_setup import setup_logging
import logging
import warnings
from dotenv import load_dotenv

load_dotenv()  # Loads .env from current directory

# Suppress all common library warnings before any imports
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Lazy import torch to avoid import errors when torch isn't available
# (e.g., when using system Python instead of venv)
try:
    import torch
    import diffusers
    from packaging import version

    _TORCH_AVAILABLE = True
except ImportError as e:
    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = e
    # Create dummy objects to prevent AttributeError
    torch = None
    diffusers = None
    version = None


__version__ = "0.37.0"

settings = load_settings()


def get_device():
    """
    Detect and return the best available device for PyTorch operations.
    Priority: CUDA > MPS > CPU

    Returns:
        str: Device identifier ('cuda', 'mps', or 'cpu')
    """
    if not _TORCH_AVAILABLE:
        return "cpu"

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
    if not _TORCH_AVAILABLE:
        raise ImportError(
            f"PyTorch is not available. {_TORCH_IMPORT_ERROR}\n"
            "Please ensure you're using the virtual environment where torch is installed.\n"
            "Activate the venv: source venv/bin/activate"
        )

    device = get_device()
    torch.set_default_device(device)

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
    logging.info(f"Using device: {device}")

    diffusers.logging.set_verbosity_error()

    # TF32 optimization (Ampere+ GPUs: RTX 30/40 series, A100, H100)
    # Provides ~2x speedup for matmul operations with minimal precision loss
    if settings.enable_tf32:
        torch.set_float32_matmul_precision("high")
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        logging.debug("TF32 precision enabled for faster matmul operations")
    else:
        logging.debug("TF32 precision disabled (full FP32 precision)")

    # CUDA-specific optimizations
    if device == "cuda":
        # cuDNN autotuner - benchmarks algorithms and selects fastest
        # Best for fixed input sizes, may slow down variable-size workflows
        torch.backends.cudnn.benchmark = settings.cudnn_benchmark

        # Always enable cuDNN (default anyway)
        torch.backends.cudnn.enabled = True

        # Deterministic mode - set True for reproducibility (same seed = same output)
        # False prioritizes performance over strict reproducibility
        torch.backends.cudnn.deterministic = settings.cudnn_deterministic

        logging.debug(
            f"CUDA optimizations - benchmark: {settings.cudnn_benchmark}, "
            f"deterministic: {settings.cudnn_deterministic}"
        )
