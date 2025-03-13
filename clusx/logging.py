"""
Logging configuration for Clusterium.

This module provides standardized logging functionality for the Clusterium package,
including configuration setup and logger retrieval. It ensures consistent log formatting
across all components of the package and simplifies the process of obtaining properly
configured logger instances.

The module offers two main functions:

- setup_logging: Configures the root logger with appropriate formatting and level
- get_logger: Returns a logger instance with the specified name

Typical usage:

    >>> from clusx.logging import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing started")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional


def setup_logging(level: Optional[int] = None) -> None:
    """
    Set up logging configuration for the application.

    This function configures the root logger with a standardized format that includes
    timestamp, log level, and message. It's typically called once at the start of
    the application to ensure consistent logging behavior across all modules.

    The timestamp format is ISO-like (YYYY-MM-DD HH:MM:SS) for better readability
    and sorting in log files.

    Args:
        level: The logging level (defaults to logging.INFO if None).
              Common levels: DEBUG(10), INFO(20), WARNING(30), ERROR(40), CRITICAL(50)
    """
    if level is None:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    This function returns a logger instance configured with the specified name,
    which is typically the module name (__name__). Using named loggers allows for
    hierarchical logging configuration and makes it easier to identify the source
    of log messages.

    The returned logger inherits settings from the root logger configured by
    setup_logging(), but can be further customized if needed.

    Args:
        name: The name for the logger (typically __name__)

    Returns:
        logging.Logger: A configured logger instance ready for use
    """
    return logging.getLogger(name)
