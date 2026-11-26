import logging
import os

from concurrent_log_handler import ConcurrentRotatingFileHandler

LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


def setup_logging(log_path, log_level="INFO", log_to_console=False):
    """Configure the 'dw' logger. Safe to call more than once - existing
    handlers are replaced, not stacked, so a reconfiguring caller (the
    server's persistent worker, configuring again after `import dw` already
    did) does not multiply every line."""
    logger = logging.getLogger("dw")
    logger.setLevel(LOG_LEVELS.get(log_level, logging.INFO))

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    file_handler = ConcurrentRotatingFileHandler(
        log_path, "a", maxBytes=50 * 1024 * 1024, backupCount=7
    )
    file_handler.setFormatter(_FORMATTER)
    logger.addHandler(file_handler)

    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(_FORMATTER)
        logger.addHandler(console_handler)

    return logger


def log_storage_stats():
    """The 'dw' logger's active log file and its rotated backups on disk
    (#1457), as `{path, size_bytes, backups, total_bytes}`.

    Read off the file handler `setup_logging` installed, so it names the
    file this process actually writes. None when no log file is configured;
    a size that can't be read is None rather than an exception."""
    handler = next(
        (
            h
            for h in logging.getLogger("dw").handlers
            if isinstance(h, logging.FileHandler) or getattr(h, "baseFilename", None)
        ),
        None,
    )
    if handler is None:
        return None
    path = os.path.abspath(handler.baseFilename)
    stats = {"path": path, "size_bytes": None, "backups": None, "total_bytes": None}
    try:
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        backup_sizes = [
            os.path.getsize(f"{path}.{n}")
            for n in range(1, int(getattr(handler, "backupCount", 0) or 0) + 1)
            if os.path.isfile(f"{path}.{n}")
        ]
    except OSError:
        logging.getLogger("dw").debug("Could not size %s", path, exc_info=True)
        return stats
    stats.update(
        size_bytes=size,
        backups=len(backup_sizes),
        total_bytes=size + sum(backup_sizes),
    )
    return stats


def set_log_level(log_level):
    """Change the 'dw' logger's level without touching its handlers."""
    logging.getLogger("dw").setLevel(LOG_LEVELS.get(log_level, logging.INFO))
