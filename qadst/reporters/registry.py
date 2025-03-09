"""Registry for managing multiple reporters."""

import logging
from typing import Dict, List

from .base import BaseReporter, ReportData

logger = logging.getLogger(__name__)


class ReporterRegistry:
    """Registry for managing multiple reporters.

    This class maintains a collection of reporters and provides methods to
    register, enable, disable, and use them for generating reports.
    """

    def __init__(self):
        """Initialize the reporter registry."""
        self.reporters: Dict[str, BaseReporter] = {}
        self.enabled_reporters: List[str] = []

    def register(self, name: str, reporter: BaseReporter, enabled: bool = True) -> None:
        """Register a reporter with the registry.

        Args:
            name: Unique name for the reporter
            reporter: Reporter instance
            enabled: Whether the reporter should be enabled by default
        """
        self.reporters[name] = reporter
        if enabled and name not in self.enabled_reporters:
            self.enabled_reporters.append(name)
        logger.debug(f"Registered reporter: {name} (enabled: {enabled})")

    def enable(self, name: str) -> None:
        """Enable a reporter.

        Args:
            name: Name of the reporter to enable
        """
        if name in self.reporters and name not in self.enabled_reporters:
            self.enabled_reporters.append(name)
            logger.debug(f"Enabled reporter: {name}")

    def disable(self, name: str) -> None:
        """Disable a reporter.

        Args:
            name: Name of the reporter to disable
        """
        if name in self.enabled_reporters:
            self.enabled_reporters.remove(name)
            logger.debug(f"Disabled reporter: {name}")

    def generate_reports(self, data: ReportData) -> None:
        """Generate reports using all enabled reporters.

        Args:
            data: Report data to be output
        """
        for name in self.enabled_reporters:
            if name in self.reporters:
                try:
                    self.reporters[name].generate_report(data)
                    logger.debug(f"Generated report using: {name}")
                except Exception as e:
                    logger.error(f"Error generating report with {name}: {e}")
            else:
                logger.warning(f"Reporter {name} is enabled but not registered")
