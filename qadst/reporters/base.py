"""Base reporter class for QA dataset clustering results."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ReportData:
    """Data structure for cluster report data.

    This class holds all the data needed for generating reports in various formats.

    Attributes:
        report_df: DataFrame containing cluster metrics and information
        clusters_json_path: Path to the clusters JSON file
        output_dir: Directory where reports should be saved
        summary_metrics: Dictionary of global metrics
        top_clusters: DataFrame containing the top N clusters by size
    """

    report_df: pd.DataFrame
    clusters_json_path: str
    output_dir: str
    summary_metrics: Dict[str, Any]
    top_clusters: pd.DataFrame


class BaseReporter(ABC):
    """Base class for all reporters.

    This abstract class defines the interface that all reporters must implement.
    Reporters are responsible for outputting clustering results and metrics in
    various formats (CSV, console, etc.).
    """

    def __init__(self, output_dir: str):
        """Initialize the reporter.

        Args:
            output_dir: Directory where reports should be saved
        """
        self.output_dir = output_dir

    @abstractmethod
    def generate_report(self, data: ReportData) -> None:
        """Generate and output the report.

        Args:
            data: Report data to be output
        """
        pass
