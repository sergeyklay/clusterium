"""Tests for the reporters module."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd

from qadst.reporters import (
    ConsoleReporter,
    CSVReporter,
    ReportData,
    ReporterRegistry,
)


class TestCSVReporter:
    """Tests for the CSVReporter class."""

    def test_init(self):
        """Test initialization of CSVReporter."""
        output_dir = "/tmp/test"
        reporter = CSVReporter(output_dir)
        assert reporter.output_dir == output_dir
        assert reporter.filename == "cluster_quality_report.csv"

    def test_generate_report(self):
        """Test generate_report method of CSVReporter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a reporter
            reporter = CSVReporter(temp_dir)

            # Create test data
            report_df = pd.DataFrame(
                {
                    "Cluster_ID": ["1", "2", "SUMMARY"],
                    "Num_QA_Pairs": [10, 20, 30],
                    "Coherence_Score": [0.8, 0.9, 0.0],
                    "Topic_Label": ["Topic 1", "Topic 2", "Summary"],
                }
            )
            data = ReportData(
                report_df=report_df,
                clusters_json_path="test.json",
                output_dir=temp_dir,
                summary_metrics={"noise_ratio": 0.1},
                top_clusters=report_df.iloc[:2],
            )

            # Generate report
            reporter.generate_report(data)

            # Check that the file was created
            output_path = os.path.join(temp_dir, "cluster_quality_report.csv")
            assert os.path.exists(output_path)

            # Check the content of the file
            df = pd.read_csv(output_path)
            assert len(df) == 3
            assert "Cluster_ID" in df.columns
            assert "Num_QA_Pairs" in df.columns
            assert "Coherence_Score" in df.columns
            assert "Topic_Label" in df.columns


class TestConsoleReporter:
    """Tests for the ConsoleReporter class."""

    def test_init(self):
        """Test initialization of ConsoleReporter."""
        output_dir = "/tmp/test"
        reporter = ConsoleReporter(output_dir)
        assert reporter.output_dir == output_dir
        assert reporter.top_n == 5

    def test_generate_report(self):
        """Test generate_report method of ConsoleReporter."""
        # Create a reporter
        reporter = ConsoleReporter("/tmp/test")

        # Create test data
        report_df = pd.DataFrame(
            {
                "Cluster_ID": ["1", "2", "SUMMARY"],
                "Num_QA_Pairs": [10, 20, 30],
                "Coherence_Score": [0.8, 0.9, 0.0],
                "Topic_Label": ["Topic 1", "Topic 2", "Summary"],
            }
        )
        data = ReportData(
            report_df=report_df,
            clusters_json_path="test.json",
            output_dir="/tmp/test",
            summary_metrics={
                "noise_ratio": 0.1,
                "davies_bouldin_score": 1.2,
                "calinski_harabasz_score": 45.6,
                "silhouette_score": 0.7,
            },
            top_clusters=report_df.iloc[:2],
        )

        # Generate report with captured stdout
        with patch("builtins.print") as mock_print:
            reporter.generate_report(data)

            # Check that print was called with the expected messages
            assert mock_print.call_count >= 10

            # Check for basic content
            mock_print.assert_any_call("Total QA pairs: 30")
            mock_print.assert_any_call("Clusters JSON: test.json")
            mock_print.assert_any_call("\nGlobal Metrics:")


class TestReporterRegistry:
    """Tests for the ReporterRegistry class."""

    def test_init(self):
        """Test initialization of ReporterRegistry."""
        registry = ReporterRegistry()
        assert registry.reporters == {}
        assert registry.enabled_reporters == []

    def test_register(self):
        """Test register method of ReporterRegistry."""
        registry = ReporterRegistry()
        reporter = MagicMock()
        registry.register("test", reporter)
        assert registry.reporters == {"test": reporter}
        assert registry.enabled_reporters == ["test"]

    def test_enable_disable(self):
        """Test enable and disable methods of ReporterRegistry."""
        registry = ReporterRegistry()
        reporter = MagicMock()
        registry.register("test", reporter, enabled=False)
        assert registry.reporters == {"test": reporter}
        assert registry.enabled_reporters == []

        registry.enable("test")
        assert registry.enabled_reporters == ["test"]

        registry.disable("test")
        assert registry.enabled_reporters == []

    def test_generate_reports(self):
        """Test generate_reports method of ReporterRegistry."""
        registry = ReporterRegistry()
        reporter1 = MagicMock()
        reporter2 = MagicMock()
        registry.register("test1", reporter1)
        registry.register("test2", reporter2)

        data = MagicMock()
        registry.generate_reports(data)

        reporter1.generate_report.assert_called_once_with(data)
        reporter2.generate_report.assert_called_once_with(data)

    def test_generate_reports_with_error(self):
        """Test generate_reports method with an error."""
        registry = ReporterRegistry()
        reporter = MagicMock()
        reporter.generate_report.side_effect = Exception("Test error")
        registry.register("test", reporter)

        data = MagicMock()
        with patch("qadst.reporters.registry.logger") as mock_logger:
            registry.generate_reports(data)
            mock_logger.error.assert_called_once()
