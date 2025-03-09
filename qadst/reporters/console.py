"""Console reporter for QA dataset clustering results."""

import logging
from typing import List, Optional

from .base import BaseReporter, ReportData

logger = logging.getLogger(__name__)


class ConsoleReporter(BaseReporter):
    """Reporter that outputs cluster metrics to the console.

    This reporter prints summary information and details about the top clusters
    to the console using direct print statements with a simple table format.
    """

    def __init__(self, output_dir: str, top_n: int = 5):
        """Initialize the console reporter.

        Args:
            output_dir: Directory where reports are saved (not used directly)
            top_n: Number of top clusters to display
        """
        super().__init__(output_dir)
        self.top_n = top_n
        self.table_width = 80

    def _print_separator(self, width: Optional[int] = None) -> None:
        """Print a separator line.

        Args:
            width: Width of the separator line
        """
        if width is None:
            width = self.table_width
        print("-" * width)

    def _print_header(self, title: str, width: Optional[int] = None) -> None:
        """Print a header with a title.

        Args:
            title: Title to display
            width: Width of the header
        """
        if width is None:
            width = self.table_width
        self._print_separator(width)
        print(f"| {title.center(width - 4)} |")
        self._print_separator(width)

    def _format_row(self, columns: List[str], widths: List[int]) -> str:
        """Format a row with columns of specified widths.

        Args:
            columns: Column values
            widths: Column widths

        Returns:
            Formatted row string
        """
        row = "| "
        for i, col in enumerate(columns):
            row += f"{str(col):<{widths[i]}} | "
        return row.rstrip()

    def generate_report(self, data: ReportData) -> None:
        """Generate a console report with cluster metrics.

        Prints summary information and details about the top clusters to the console
        using a simple table format.

        Args:
            data: Report data to be output
        """
        # Define column widths for consistent formatting
        col_widths = [10, 12, 12, 40]
        # Calculate total table width including borders and padding
        self.table_width = sum(col_widths) + (len(col_widths) * 3) + 1

        # Print summary information
        summary_row = data.report_df.iloc[-1]

        self._print_header("CLUSTER ANALYSIS SUMMARY")
        print(f"Total QA pairs: {summary_row['Num_QA_Pairs']}")
        print(f"Clusters JSON: {data.clusters_json_path}")

        # Format metrics string
        metrics = data.summary_metrics
        print("\nGlobal Metrics:")
        print(f"  Noise Ratio: {metrics.get('noise_ratio', 'N/A'):.2f}")
        print(
            f"  Davies-Bouldin Index: {metrics.get('davies_bouldin_score', 'N/A'):.2f}"
        )
        ch_score = metrics.get("calinski_harabasz_score", "N/A")
        print(f"  Calinski-Harabasz Index: {ch_score:.2f}")
        if "silhouette_score" in metrics:
            print(f"  Silhouette Score: {metrics.get('silhouette_score', 'N/A'):.2f}")

        # Print information about top clusters
        self._print_header(f"TOP {self.top_n} CLUSTERS BY SIZE")

        # Print table header
        header = self._format_row(
            ["Cluster ID", "Size", "Coherence", "Topic"], col_widths
        )
        print(header)
        self._print_separator()

        # Print table rows
        for _, row in data.top_clusters.iterrows():
            cluster_id = row["Cluster_ID"]
            size = row["Num_QA_Pairs"]
            coherence = f"{row['Coherence_Score']:.2f}"
            topic = row["Topic_Label"]

            # Truncate topic if too long
            if len(topic) > col_widths[3] - 3:
                topic = topic[: col_widths[3] - 6] + "..."

            print(self._format_row([cluster_id, size, coherence, topic], col_widths))

        self._print_separator()
        print()
