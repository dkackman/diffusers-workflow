from .settings import resolve_path, load_settings
from .log_setup import setup_logging
import os
import re
import logging
import warnings
from dotenv import load_dotenv

load_dotenv()  # Loads .env from current directory

# Allow MPS to use all available unified memory by default
# This prevents "MPS backend out of memory" errors with large models
# Users can override by setting these env vars before import
if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Let the CUDA allocator grow segments instead of fragmenting fixed-size ones.
# Multi-step workflows churn differently-shaped allocations (generate, upscale,
# interpolate), and fragmentation is what OOMs a card that nominally has room
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Load sharded checkpoints in parallel - a pure cold-start win
if "HF_ENABLE_PARALLEL_LOADING" not in os.environ:
    os.environ["HF_ENABLE_PARALLEL_LOADING"] = "true"

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


def _read_version():
    # Single-sourced from pyproject.toml. Read as text rather than TOML:
    # tomllib is 3.11+ and this project supports 3.10. In a wheel install
    # there is no pyproject on disk - the package metadata carries it.
    try:
        pyproject = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
        with open(pyproject, encoding="utf-8") as f:
            match = re.search(r'^version = "([^"]+)"$', f.read(), re.MULTILINE)
        if match:
            return match.group(1)
    except OSError:
        pass
    try:
        from importlib.metadata import version

        return version("diffusers-workflow")
    except Exception:
        return "unknown"


__version__ = _read_version()

settings = load_settings()


def detect_device():
    """
    Detect the best device available on this machine.
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


def get_device():
    """
    Get the device to run on.

    An explicit choice wins over detection - the DW_DEVICE environment variable for a
    single run, or the 'device' setting for a standing one. Either can name a specific
    accelerator ('cuda:1'), force a CPU run to rule out a GPU-specific problem, or pin
    a machine that has both to one backend.

    Returns:
        str: Device identifier ('cuda', 'cuda:1', 'mps', 'cpu', ...)
    """
    if not _TORCH_AVAILABLE:
        return "cpu"

    configured_device = os.environ.get("DW_DEVICE") or settings.device
    if configured_device:
        try:
            # Only checks that torch understands the name - whether the machine has
            # that device surfaces when a model is loaded onto it
            torch.device(configured_device)
            return configured_device
        except (RuntimeError, TypeError, ValueError) as e:
            logging.warning(f"Ignoring invalid configured device: {e}")

    return detect_device()


def get_device_type(device=None):
    """
    Get the backend a device belongs to - 'cuda' for 'cuda:1', 'mps' for 'mps'.

    A device identifier can carry an index, so comparisons against a backend name have
    to be made on the type alone or a device like 'cuda:1' matches nothing.

    Args:
        device: Device identifier, or None to use the device dw is running on

    Returns:
        str: Backend name ('cuda', 'mps', 'xpu', 'cpu', ...)
    """
    if not _TORCH_AVAILABLE:
        return "cpu"

    return torch.device(device if device is not None else get_device()).type


def backend_available(device_type):
    """
    Whether this machine can actually run on a backend.

    Only the backends dw knows how to probe get an answer. An unrecognized one is
    reported available rather than guessed at, so a device torch understands and dw
    does not is left alone instead of being rewritten.

    Args:
        device_type: Backend name ('cuda', 'mps', 'cpu', ...)

    Returns:
        bool: Whether the backend is usable here
    """
    if not _TORCH_AVAILABLE:
        return device_type == "cpu"

    if device_type == "cpu":
        return True

    if device_type == "cuda":
        return torch.cuda.is_available()

    if device_type == "mps":
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    is_available = getattr(getattr(torch, device_type, None), "is_available", None)
    return bool(is_available()) if callable(is_available) else True


def resolve_device(requested):
    """
    Translate a device a workflow asked for into one this machine has.

    A device named in a workflow is nearly always the author saying 'the accelerator'
    rather than 'specifically NVIDIA', so a workflow written on a CUDA box runs on a
    Mac and back again. The backend is what gets translated - an index is left alone
    when the backend matches, since 'cuda:1' on a single-GPU CUDA box is a genuine
    mistake and not a portability problem. A CPU device is never rewritten: pinning a
    step to the CPU is how a GPU-specific problem gets ruled out.

    Args:
        requested: Device identifier from a workflow, or None for no override

    Returns:
        The device to use, or None if None was passed
    """
    if requested is None or not _TORCH_AVAILABLE:
        return requested

    try:
        device = torch.device(requested)
    except (RuntimeError, TypeError, ValueError):
        # Not a device torch understands - let the failure surface where it is used
        return requested

    if device.type == "cpu" or backend_available(device.type):
        return requested

    target = get_device()
    if device.index is not None:
        logging.warning(
            f"Workflow asks for '{requested}', which this machine has no {device.type} "
            f"backend for - running on {target}. The device index is dropped, so a "
            "workflow that meant to spread work across accelerators will not."
        )
    else:
        logging.warning(
            f"Workflow asks for '{requested}', which this machine has no {device.type} "
            f"backend for - running on {target}"
        )

    return target


def get_autocast_device_type():
    """
    Get the device type to use for torch.autocast.
    MPS doesn't support autocast, so use 'cpu' for MPS devices.

    Returns:
        str: Device type for autocast ('cuda' or 'cpu')
    """
    if get_device_type() == "cuda":
        return "cuda"

    # MPS and CPU both use 'cpu' for autocast
    return "cpu"


def preferred_task_dtype(device=None):
    """
    Get the preferred torch dtype for a lightweight inference task (captioning,
    text generation, diffusion upscaling, depth estimation).

    fp16 produces NaN values on MPS and is unsupported by many CPU operations,
    so only CUDA gets it - everything else runs in fp32.

    Args:
        device: Device identifier, or None to use the device dw is running on

    Returns:
        torch.dtype: torch.float16 on CUDA, torch.float32 otherwise (or None
            if torch isn't available)
    """
    if not _TORCH_AVAILABLE:
        return None

    return torch.float16 if get_device_type(device) == "cuda" else torch.float32


def empty_device_cache(synchronize=False):
    """
    Empty the allocator cache of the device dw is configured for, if its
    backend has one. A no-op on CPU.

    Dispatch honors the DW_DEVICE/settings override (via get_device_type())
    but also checks the backend's own availability before calling into it,
    matching the belt-and-suspenders checks the call sites used to do
    individually.

    Args:
        synchronize: Also block until pending device work completes before
            returning. Skipped by default - it's expensive and callers on a
            hot path (e.g. between-run cleanup) don't need it.
    """
    if not _TORCH_AVAILABLE:
        return

    device_type = get_device_type()
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        if synchronize:
            torch.cuda.synchronize()
    elif (
        device_type == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        if synchronize and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


def device_memory_stats():
    """
    Snapshot of allocator memory usage for the device dw is configured for.

    Only CUDA exposes free/total figures (via torch.cuda.mem_get_info); MPS
    has no equivalent API, so its snapshot reports zeroed, "known" values
    instead. CPU, or a CUDA/MPS backend that isn't actually available despite
    being the configured device type, results in an "unavailable" snapshot.

    Returns:
        dict with keys:
            available (bool): whether the backend could be queried
            device_name (str or None)
            allocated_mb (float)
            reserved_mb (float)
            free_mb (float or None): None only when CUDA's mem_get_info call
                itself fails
            total_mb (float or None): None only when CUDA's mem_get_info call
                itself fails
    """
    stats = {
        "available": False,
        "device_name": None,
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "free_mb": None,
        "total_mb": None,
    }

    if not _TORCH_AVAILABLE:
        return stats

    device_type = get_device_type()
    if device_type == "cuda" and torch.cuda.is_available():
        stats["available"] = True
        stats["device_name"] = torch.cuda.get_device_name(0)
        stats["allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        stats["reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        try:
            free, total = torch.cuda.mem_get_info()
            stats["free_mb"] = free / 1024 / 1024
            stats["total_mb"] = total / 1024 / 1024
        except (RuntimeError, AttributeError):
            pass
    elif (
        device_type == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        # MPS doesn't provide detailed memory stats like CUDA
        stats["available"] = True
        stats["device_name"] = "Apple Silicon (MPS)"
        stats["free_mb"] = 0.0
        stats["total_mb"] = 0.0

    return stats


def startup(log_level=None):
    if not _TORCH_AVAILABLE:
        raise ImportError(
            f"PyTorch is not available. {_TORCH_IMPORT_ERROR}\n"
            "Please ensure you're using the virtual environment where torch is installed.\n"
            "Activate the venv: source venv/bin/activate"
        )

    # The default torch device is deliberately left alone. Diffusers loads weights into
    # system memory and then places them - moving them to the device, or hooking them
    # for offloading. A default device of 'cuda' pre-empts that by building every module
    # directly in VRAM, which runs a large pipeline out of memory before its offload
    # hooks are ever installed. Device placement is explicit throughout dw instead.
    device = get_device()
    device_type = get_device_type(device)

    # MPS-specific configuration (Apple Silicon)
    if device_type == "mps":
        # Suppress autocast warnings (MPS doesn't support autocast,
        # and libraries may try to use it with 'cuda' device_type)
        warnings.filterwarnings(
            "ignore",
            message=".*User provided device_type of 'cuda'.*",
            category=UserWarning,
            module="torch.amp.autocast_mode",
        )

        logging.info(
            f"MPS backend - watermark ratio: "
            f"{os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'default')}"
        )

    # Check if we have a GPU backend (CUDA or MPS)
    if device_type == "cpu":
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
        log_to_console=settings.log_to_console,
    )

    logging.info(f"Version {__version__}")
    logging.debug(f"Torch version {torch.__version__}")
    logging.info(f"Using device: {device}")

    diffusers.logging.set_verbosity_error()

    # TF32 optimization (Ampere+ GPUs: RTX 30/40 series, A100, H100)
    # Provides ~2x speedup for matmul operations with minimal precision loss
    if settings.enable_tf32:
        torch.set_float32_matmul_precision("high")
        if device_type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        logging.debug("TF32 precision enabled for faster matmul operations")
    else:
        logging.debug("TF32 precision disabled (full FP32 precision)")

    # CUDA-specific optimizations
    if device_type == "cuda":
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
