"""
Logger module for Assam Folk Music Classification project.
Provides structured colored console logging and file logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "AssamFolkMusic",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configures and returns a logging logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
