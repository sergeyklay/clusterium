"""CSV reporter for QA dataset clustering results."""

import logging
from pathlib import Path

from .base import BaseReporter, ReportData

logger = logging.getLogger(__name__)


class CSVReporter(BaseReporter):
    """Reporter that outputs cluster metrics to a CSV file.

    This reporter generates a CSV file containing detailed information about
    each cluster, including size, coherence scores, and topic labels.
    """

    def __init__(self, output_dir: str, filename: str = "cluster_quality_report.csv"):
        """Initialize the CSV reporter.

        Args:
            output_dir: Directory where the CSV file should be saved
            filename: Name of the CSV file to create
        """
        super().__init__(output_dir)
        self.filename = filename

    def generate_report(self, data: ReportData) -> None:
        """Generate a CSV report with cluster metrics.

        Saves a CSV file containing detailed information about each cluster,
        including a summary row with global metrics.

        Args:
            data: Report data to be output
        """
        output_path = Path(self.output_dir) / self.filename
        data.report_df.to_csv(output_path, index=False)
        logger.debug(f"CSV report saved to: {output_path}")
