import logging
import logging.handlers
import os


def setup_logging(log_name: str, max_bytes: int = 5 * 1024 * 1024, backup_count: int = 3) -> logging.Logger:
    """
    Configure the root logger with a rotating file handler.

    All child loggers (logging.getLogger(__name__) in algos/, utils/, worker_task.py)
    propagate to root automatically — no extra wiring needed.

    Args:
        log_name:     Base name for the log file (no extension). E.g. "run_association_mining_fpgrowth".
        max_bytes:    Max size per log file before rotation. Default 5 MB.
        backup_count: Number of rotated files to keep. Default 3.

    Returns:
        Root logger.
    """
    log_path = os.path.join(os.getcwd(), f"{log_name}.log")

    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s - %(message)s"))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)

    return root
