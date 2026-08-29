import logging
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
    handlers are replaced, not stacked, so a reconfiguring caller (the REPL
    worker between runs) does not multiply every line."""
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


def set_log_level(log_level):
    """Change the 'dw' logger's level without touching its handlers."""
    logging.getLogger("dw").setLevel(LOG_LEVELS.get(log_level, logging.INFO))
