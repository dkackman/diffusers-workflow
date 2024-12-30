import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler


def setup_logging(log_path, log_level="INFO"):
    # Use a dictionary to map log level strings to log level constants
    log_levels = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }

    logger = logging.getLogger("dh")
    logger.setLevel(log_levels.get(log_level, logging.INFO))

    file_handler = ConcurrentRotatingFileHandler(
        log_path, "a", maxBytes=50 * 1024 * 1024, backupCount=7
    )
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    return logger
