import logging
import logging.handlers
import os
import shutil
import datetime
import json


def archive_previous_run(results_algo_dir: str, results_base_dir: str, algo: str):
    """
    Archives the existing result directory by appending a timestamp to its name.
    Then ensures a fresh directory is created.
    """
    if os.path.exists(results_algo_dir):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        archive_dir = os.path.join(results_base_dir, f"{algo}_ARCHIVE_{timestamp}")
        shutil.move(results_algo_dir, archive_dir)

    os.makedirs(results_algo_dir, exist_ok=True)


def save_run_config(results_algo_dir: str, run_config: dict, logger: logging.Logger):
    """
    Saves the run configuration to a JSON file and prints it to the logger.
    """
    with open(os.path.join(results_algo_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=4)
        
    logger.info("--- RUN CONFIGURATION ---")
    for k, v in run_config.items():
        if isinstance(v, dict):
            logger.info(f"  {k}:")
            for sub_k, sub_v in v.items():
                logger.info(f"    {sub_k}: {sub_v}")
        else:
            logger.info(f"  {k}: {v}")
    logger.info("-------------------------")


def setup_logging(log_name: str, max_bytes: int = 5 * 1024 * 1024, backup_count: int = 3, log_dir: str = None) -> logging.Logger:
    """
    Configure the root logger with a rotating file handler.

    All child loggers (logging.getLogger(__name__) in algos/, utils/, worker_task.py)
    propagate to root automatically — no extra wiring needed.

    Args:
        log_name:     Base name for the log file (no extension). E.g. "run_association_mining_fpgrowth".
        max_bytes:    Max size per log file before rotation. Default 5 MB.
        backup_count: Number of rotated files to keep. Default 3.
        log_dir:      Directory to save the log file. Default is current working directory.

    Returns:
        Root logger.
    """
    if log_dir is None:
        log_dir = os.getcwd()
    else:
        os.makedirs(log_dir, exist_ok=True)
        
    log_path = os.path.join(log_dir, f"{log_name}.log")

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
