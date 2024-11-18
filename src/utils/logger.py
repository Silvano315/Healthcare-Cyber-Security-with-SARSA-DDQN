import logging
import os
from typing import Optional
from datetime import datetime

def setup_logger(
    name: str,
    level: str = "INFO",
    log_dir: str = "logs",
    filename: Optional[str] = None
) -> logging.Logger:
    """
    Set up logger with both file and console handlers.
    
    Args:
        name: Name of the logger
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        filename: Specific filename for the log file. If None, generates timestamp-based name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    logger.handlers = []

    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    os.makedirs(log_dir, exist_ok=True)
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name.lower()}_{timestamp}.log"

    file_handler = logging.FileHandler(
        os.path.join(log_dir, filename)
    )
    file_handler.setLevel(getattr(logging, level.upper()))
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger
