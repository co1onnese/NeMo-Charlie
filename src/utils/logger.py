"""
logger.py
Centralized logging configuration for the SFT pipeline.
Provides colorized console output and file logging with rotation.
"""
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
    file_logging: bool = True
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    Args:
        name: Logger name (usually __name__)
        log_dir: Directory for log files (default: logs/)
        log_file: Specific log file name (default: auto-generated with timestamp)
        level: Logging level
        console: Whether to log to console
        file_logging: Whether to log to file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear any existing handlers
    
    # Create formatters
    if HAS_COLORLOG and console:
        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s%(reset)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    else:
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_logging:
        if log_dir is None:
            log_dir = os.environ.get("LOG_DIR", "logs")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"{name.replace('.', '_')}_{timestamp}.log"
        
        log_path = os.path.join(log_dir, log_file)
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_path}")
    
    return logger


def get_logger(name: str, **kwargs) -> logging.Logger:
    """
    Convenience function to get or create a logger.
    
    Args:
        name: Logger name
        **kwargs: Additional arguments passed to setup_logger
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger = setup_logger(name, **kwargs)
    return logger
