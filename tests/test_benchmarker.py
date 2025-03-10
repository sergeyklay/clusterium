"""Tests for the benchmarker module."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qadst.benchmarker import ClusterBenchmarker


class TestClusterBenchmarker:
    """Tests for the ClusterBenchmarker class."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        mock_embeddings_provider = MagicMock()
        benchmarker = ClusterBenchmarker(embeddings_provider=mock_embeddings_provider)
        assert benchmarker.embeddings_provider == mock_embeddings_provider
        assert benchmarker.llm is None
        assert benchmarker.output_dir == "./output"

    def test_init_with_models(self):
        """Test initialization with model names."""
        # Create mocks
        mock_embeddings_provider = MagicMock()
        mock_llm = MagicMock()

        # Setup patches
        with (
            patch(
                "qadst.benchmarker.ChatOpenAI", return_value=mock_llm
            ) as mock_chat_openai,
            patch("qadst.benchmarker.ReporterRegistry"),  # Mock the reporter registry
        ):
            # Create benchmarker with model names
            benchmarker = ClusterBenchmarker(
                embeddings_provider=mock_embeddings_provider,
                llm_model_name="test-llm-model",
                output_dir="/tmp/test",
            )

            # Check that models were initialized correctly
            assert benchmarker.embeddings_provider == mock_embeddings_provider
            assert benchmarker.llm == mock_llm
            assert benchmarker.output_dir == "/tmp/test"

            # Check that the models were created with the right parameters
            mock_chat_openai.assert_called_once_with(
                model="test-llm-model", temperature=0.0
            )

    def test_init_with_model_errors(self):
        """Test initialization with model errors."""
        # Create mock
        mock_embeddings_provider = MagicMock()

        # Setup patches with exceptions
        with (
            patch("qadst.benchmarker.ChatOpenAI", side_effect=Exception("LLM error")),
            patch("qadst.benchmarker.ReporterRegistry"),  # Mock the reporter registry
            patch("qadst.benchmarker.logger") as mock_logger,  # Mock the logger
        ):
            # Create benchmarker with model names
            benchmarker = ClusterBenchmarker(
                embeddings_provider=mock_embeddings_provider,
                llm_model_name="test-llm-model",
            )

            # Check that models are None due to errors
            assert benchmarker.embeddings_provider == mock_embeddings_provider
            assert benchmarker.llm is None

            # Verify warning logs were created
            mock_logger.warning.assert_any_call("Failed to initialize LLM: LLM error")

    @patch("qadst.embeddings.get_embeddings_model")
    def test_load_clusters_valid_json(self, mock_embeddings_model):
        """Test loading clusters from a valid JSON file."""
        mock_embeddings_provider = MagicMock()
        benchmarker = ClusterBenchmarker(embeddings_provider=mock_embeddings_provider)

        # Create a temporary JSON file with valid content
        clusters_data = {
            "clusters": {
                "0": ["Q1", "Q2"],
                "1": ["Q3", "Q4", "Q5"],
            }
        }
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            json.dump(clusters_data, f)
            json_path = f.name

        try:
            # Load the clusters
            result = benchmarker.load_clusters(json_path)

            # Check that the clusters were loaded correctly
            assert result == clusters_data
        finally:
            # Clean up
            if os.path.exists(json_path):
                os.unlink(json_path)

    @patch("qadst.embeddings.get_embeddings_model")
    def test_load_clusters_invalid_json(self, mock_embeddings_model):
        """Test loading clusters from an invalid JSON file."""
        mock_embeddings_provider = MagicMock()
        benchmarker = ClusterBenchmarker(embeddings_provider=mock_embeddings_provider)

        # Create a temporary file with invalid JSON content
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            f.write("This is not valid JSON")
            json_path = f.name

        try:
            # Try to load the clusters
            with pytest.raises(json.JSONDecodeError):
                benchmarker.load_clusters(json_path)
        finally:
            # Clean up
            if os.path.exists(json_path):
                os.unlink(json_path)

    @patch("qadst.embeddings.get_embeddings_model")
    def test_load_clusters_file_not_found(self, mock_embeddings_model):
        """Test loading clusters from a non-existent file."""
        mock_embeddings_provider = MagicMock()
        benchmarker = ClusterBenchmarker(embeddings_provider=mock_embeddings_provider)

        # Try to load clusters from a non-existent file
        with pytest.raises(FileNotFoundError):
            benchmarker.load_clusters("non_existent_file.json")

    @patch("qadst.embeddings.get_embeddings_model")
    def test_load_qa_pairs_valid_csv(self, mock_embeddings_model):
        """Test loading QA pairs from a valid CSV file."""
        mock_embeddings_provider = MagicMock()
        benchmarker = ClusterBenchmarker(embeddings_provider=mock_embeddings_provider)

        # Create a temporary CSV file with valid content
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as f:
            f.write("question,answer\n")
            f.write("Q1,A1\n")
            f.write("Q2,A2\n")
            f.write("Q3,A3\n")
            csv_path = f.name

        try:
            # Load the QA pairs
            qa_pairs = benchmarker.load_qa_pairs(csv_path)

            # Check that the QA pairs were loaded correctly
            assert len(qa_pairs) == 3
            assert qa_pairs[0] == ("Q1", "A1")
            assert qa_pairs[1] == ("Q2", "A2")
            assert qa_pairs[2] == ("Q3", "A3")
        finally:
            # Clean up
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    @patch("qadst.embeddings.get_embeddings_model")
    def test_load_qa_pairs_empty_csv(self, mock_embeddings_model):
        """Test loading QA pairs from an empty CSV file."""
        mock_embeddings_provider = MagicMock()
        benchmarker = ClusterBenchmarker(embeddings_provider=mock_embeddings_provider)

        # Create a temporary CSV file with only headers
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as f:
            f.write("question,answer\n")
            csv_path = f.name

        try:
            # Load the QA pairs
            qa_pairs = benchmarker.load_qa_pairs(csv_path)

            # Check that an empty list was returned
            assert len(qa_pairs) == 0
        finally:
            # Clean up
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    @patch("qadst.embeddings.get_embeddings_model")
    def test_load_qa_pairs_malformed_csv(self, mock_embeddings_model):
        """Test loading QA pairs from a malformed CSV file."""
        mock_embeddings_provider = MagicMock()
        benchmarker = ClusterBenchmarker(embeddings_provider=mock_embeddings_provider)

        # Create a temporary CSV file with malformed content (missing column)
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as f:
            f.write("question,answer\n")
            f.write("Q1\n")  # Missing answer column
            csv_path = f.name

        try:
            # Try to load the QA pairs
            with pytest.raises(ValueError, match="Row does not have enough columns"):
                benchmarker.load_qa_pairs(csv_path)
        finally:
            # Clean up
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    @patch("qadst.embeddings.get_embeddings_model")
    def test_load_qa_pairs_file_not_found(self, mock_embeddings_model):
        """Test loading QA pairs from a non-existent file."""
        mock_embeddings_provider = MagicMock()
        benchmarker = ClusterBenchmarker(embeddings_provider=mock_embeddings_provider)

        # Try to load QA pairs from a non-existent file
        with pytest.raises(FileNotFoundError):
            benchmarker.load_qa_pairs("non_existent_file.csv")

    @patch("qadst.embeddings.get_embeddings_model")
    def test_extract_embeddings_from_qa_pairs_success(self, mock_embeddings_model):
        """Test extracting embeddings from QA pairs."""
        # Setup mock
        mock_embeddings_instance = MagicMock()
        mock_embeddings_model.return_value = mock_embeddings_instance

        # Create mock embeddings provider
        mock_embeddings_provider = MagicMock()
        mock_embeddings_provider.get_embeddings = MagicMock(
            return_value=[
                np.array([0.1, 0.2]),
                np.array([0.3, 0.4]),
                np.array([0.5, 0.6]),
            ]
        )

        # Create benchmarker with a mock embeddings provider
        benchmarker = ClusterBenchmarker(embeddings_provider=mock_embeddings_provider)

        # Extract embeddings
        qa_pairs = [("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3")]
        embeddings = benchmarker.extract_embeddings_from_qa_pairs(qa_pairs)

        # Check that embeddings were extracted correctly
        assert len(embeddings) == 3
        assert np.array_equal(embeddings[0], np.array([0.1, 0.2]))
        assert np.array_equal(embeddings[1], np.array([0.3, 0.4]))
        assert np.array_equal(embeddings[2], np.array([0.5, 0.6]))

        # Check that get_embeddings was called with the right parameters
        mock_embeddings_provider.get_embeddings.assert_called_once_with(
            ["Q1", "Q2", "Q3"]
        )

    @patch("qadst.embeddings.get_embeddings_model")
    def test_extract_embeddings_from_qa_pairs_empty_list(self, mock_embeddings_model):
        """Test extracting embeddings from an empty list of QA pairs."""
        # Setup mock
        mock_embeddings_instance = MagicMock()
        mock_embeddings_model.return_value = mock_embeddings_instance

        # Create mock embeddings provider
        mock_embeddings_provider = MagicMock()
        mock_embeddings_provider.get_embeddings = MagicMock(return_value=[])

        # Create benchmarker with a mock embeddings provider
        benchmarker = ClusterBenchmarker(embeddings_provider=mock_embeddings_provider)

        # Extract embeddings from an empty list
        qa_pairs = []
        embeddings = benchmarker.extract_embeddings_from_qa_pairs(qa_pairs)

        # Check that an empty array was returned
        assert len(embeddings) == 0
        assert isinstance(embeddings, np.ndarray)

        # Check that get_embeddings was called with an empty list
        mock_embeddings_provider.get_embeddings.assert_called_once_with([])

    @patch("qadst.embeddings.get_embeddings_model")
    def test_extract_embeddings_from_qa_pairs_model_error(self, mock_embeddings_model):
        """Test extracting embeddings with a model error."""
        # Setup mock
        mock_embeddings_instance = MagicMock()
        mock_embeddings_model.return_value = mock_embeddings_instance

        # Create mock embeddings provider that raises an exception
        mock_embeddings_provider = MagicMock()
        mock_embeddings_provider.get_embeddings = MagicMock(
            side_effect=Exception("Model error")
        )

        # Create benchmarker with a mock embeddings provider
        benchmarker = ClusterBenchmarker(embeddings_provider=mock_embeddings_provider)

        # Try to extract embeddings
        qa_pairs = [("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3")]
        with pytest.raises(Exception, match="Model error"):
            benchmarker.extract_embeddings_from_qa_pairs(qa_pairs)
