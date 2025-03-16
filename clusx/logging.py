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

    Parameters
    ----------
    name : str
        The name for the logger (typically ``__name__``).

    Returns
    -------
    logging.Logger
        A configured logger instance ready for use.
    """
    return logging.getLogger(name)
