"""Logging configuration utilities for the ai_essentials project.

This module provides a centralized logging setup function that creates
named loggers with both file and console handlers. Log files are stored
in the outputs/logs directory.
"""

import logging
from pathlib import Path
from config.paths import OUTPUTS_DIR


def setup_logging(name: str="rag_assistant"):
    """Create and configure a named logger with file and console handlers.

    Creates a logger that writes INFO+ messages to a log file and ERROR+
    messages to the console. Log files are stored in outputs/logs/ with the
    logger name as the filename. Returns existing logger if already configured.

    Args:
        name: Logger name, used for both the logger instance and log filename.

    Returns:
        Configured logging.Logger instance.
    """
    log_path = Path(OUTPUTS_DIR) / f"logs/{name}.log"

    # Ensure the logs directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger
    
    formatter = logging.Formatter('\n-------------\n%(asctime)s - %(levelname)s - [Line: %(lineno)d] -\n%(message)s')

    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.ERROR)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger