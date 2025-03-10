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

    @patch("qadst.embeddings.get_embeddings_model")
    def test_init_default_values(self, mock_embeddings_model):
        """Test initialization with default values."""
        benchmarker = ClusterBenchmarker()
        assert benchmarker.embeddings_provider is None
        assert benchmarker.llm is None
        assert benchmarker.output_dir == "./output"

    def test_init_with_models(self):
        """Test initialization with model names."""
        # Create mocks
        mock_embeddings_model = MagicMock()
        mock_embeddings_provider = MagicMock()
        mock_llm = MagicMock()

        # Setup patches
        with (
            patch(
                "qadst.benchmarker.get_embeddings_model",
                return_value=mock_embeddings_model,
            ) as mock_get_embeddings,
            patch(
                "qadst.benchmarker.EmbeddingsProvider",
                return_value=mock_embeddings_provider,
            ) as mock_provider_class,
            patch(
                "qadst.benchmarker.ChatOpenAI", return_value=mock_llm
            ) as mock_chat_openai,
            patch("qadst.benchmarker.ReporterRegistry"),  # Mock the reporter registry
        ):
            # Create benchmarker with model names
            benchmarker = ClusterBenchmarker(
                embedding_model_name="test-embedding-model",
                llm_model_name="test-llm-model",
                output_dir="/tmp/test",
            )

            # Check that models were initialized correctly
            assert benchmarker.embeddings_provider == mock_embeddings_provider
            assert benchmarker.llm == mock_llm
            assert benchmarker.output_dir == "/tmp/test"

            # Check that the models were created with the right parameters
            mock_get_embeddings.assert_called_once_with(
                model_name="test-embedding-model"
            )
            mock_provider_class.assert_called_once_with(
                model=mock_embeddings_model, output_dir="/tmp/test"
            )
            mock_chat_openai.assert_called_once_with(
                model="test-llm-model", temperature=0.0
            )

    def test_init_with_model_errors(self):
        """Test initialization with model errors."""
        # Setup patches with exceptions
        with (
            patch(
                "qadst.benchmarker.get_embeddings_model",
                side_effect=Exception("Embeddings model error"),
            ),
            patch("qadst.benchmarker.ChatOpenAI", side_effect=Exception("LLM error")),
            patch("qadst.benchmarker.ReporterRegistry"),  # Mock the reporter registry
            patch("qadst.benchmarker.logger") as mock_logger,  # Mock the logger
        ):
            # Create benchmarker with model names
            benchmarker = ClusterBenchmarker(
                embedding_model_name="test-embedding-model",
                llm_model_name="test-llm-model",
            )

            # Check that models are None due to errors
            assert benchmarker.embeddings_provider is None
            assert benchmarker.llm is None

            # Verify warning logs were created
            mock_logger.warning.assert_any_call(
                "Failed to initialize embeddings model: Embeddings model error"
            )
            mock_logger.warning.assert_any_call("Failed to initialize LLM: LLM error")

    @patch("qadst.embeddings.get_embeddings_model")
    def test_load_clusters_valid_json(self, mock_embeddings_model):
        """Test loading clusters from a valid JSON file."""
        benchmarker = ClusterBenchmarker()

        # Create a temporary JSON file with valid clusters
        clusters_data = {
            "clusters": [
                {
                    "id": 1,
                    "representative": [{"question": "Q1", "answer": "A1"}],
                    "source": ["source1"],
                },
                {
                    "id": 2,
                    "representative": [{"question": "Q2", "answer": "A2"}],
                    "source": ["source2"],
                },
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(clusters_data, f)
            json_path = f.name

        try:
            # Load clusters from the file
            result = benchmarker.load_clusters(json_path)

            # Check that the clusters were loaded correctly
            assert result == clusters_data
        finally:
            # Clean up the temporary file
            os.unlink(json_path)

    @patch("qadst.embeddings.get_embeddings_model")
    def test_load_clusters_invalid_json(self, mock_embeddings_model):
        """Test loading clusters from an invalid JSON file."""
        benchmarker = ClusterBenchmarker()

        # Create a temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("This is not valid JSON")
            json_path = f.name

        try:
            # Try to load clusters from the file
            with pytest.raises(json.JSONDecodeError):
                benchmarker.load_clusters(json_path)
        finally:
            # Clean up the temporary file
            os.unlink(json_path)

    @patch("qadst.embeddings.get_embeddings_model")
    def test_load_clusters_file_not_found(self, mock_embeddings_model):
        """Test loading clusters from a non-existent file."""
        benchmarker = ClusterBenchmarker()

        # Try to load clusters from a non-existent file
        with pytest.raises(FileNotFoundError):
            benchmarker.load_clusters("/path/to/nonexistent/file.json")

    @patch("qadst.embeddings.get_embeddings_model")
    def test_load_qa_pairs_valid_csv(self, mock_embeddings_model):
        """Test loading QA pairs from a valid CSV file."""
        benchmarker = ClusterBenchmarker()

        # Create a temporary CSV file with valid QA pairs
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("question,answer\n")
            f.write("Q1,A1\n")
            f.write("Q2,A2\n")
            csv_path = f.name

        try:
            # Load QA pairs from the file
            result = benchmarker.load_qa_pairs(csv_path)

            # Check that the QA pairs were loaded correctly
            assert len(result) == 2
            assert result[0] == ("Q1", "A1")
            assert result[1] == ("Q2", "A2")
        finally:
            # Clean up the temporary file
            os.unlink(csv_path)

    @patch("qadst.embeddings.get_embeddings_model")
    def test_load_qa_pairs_empty_csv(self, mock_embeddings_model):
        """Test loading QA pairs from an empty CSV file."""
        benchmarker = ClusterBenchmarker()

        # Create a temporary CSV file with only the header
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("question,answer\n")
            csv_path = f.name

        try:
            # Load QA pairs from the file
            result = benchmarker.load_qa_pairs(csv_path)

            # Check that the result is an empty list
            assert result == []
        finally:
            # Clean up the temporary file
            os.unlink(csv_path)

    @patch("qadst.embeddings.get_embeddings_model")
    def test_load_qa_pairs_malformed_csv(self, mock_embeddings_model):
        """Test loading QA pairs from a malformed CSV file."""
        benchmarker = ClusterBenchmarker()

        # Create a temporary CSV file with a malformed row
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("question,answer\n")
            f.write("Q1,A1\n")
            f.write("Q2\n")  # Missing answer
            csv_path = f.name

        try:
            # Load QA pairs from the file
            with pytest.raises(ValueError):
                benchmarker.load_qa_pairs(csv_path)
        finally:
            # Clean up the temporary file
            os.unlink(csv_path)

    @patch("qadst.embeddings.get_embeddings_model")
    def test_load_qa_pairs_file_not_found(self, mock_embeddings_model):
        """Test loading QA pairs from a non-existent file."""
        benchmarker = ClusterBenchmarker()

        # Try to load QA pairs from a non-existent file
        with pytest.raises(FileNotFoundError):
            benchmarker.load_qa_pairs("/path/to/nonexistent/file.csv")

    @patch("qadst.embeddings.get_embeddings_model")
    def test_extract_embeddings_from_qa_pairs_success(self, mock_embeddings_model):
        """Test extracting embeddings from QA pairs."""
        # Setup mock
        mock_embeddings_instance = MagicMock()
        mock_embeddings_model.return_value = mock_embeddings_instance

        # Create benchmarker with a mock embeddings model
        benchmarker = ClusterBenchmarker(embedding_model_name="test-model")

        # Create a mock embeddings provider
        benchmarker.embeddings_provider = MagicMock()

        # Mock the get_embeddings method
        mock_embeddings = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
        benchmarker.embeddings_provider.get_embeddings = MagicMock(
            return_value=mock_embeddings
        )

        # Extract embeddings from QA pairs
        qa_pairs = [("Q1", "A1"), ("Q2", "A2")]
        result = benchmarker.extract_embeddings_from_qa_pairs(qa_pairs)

        # Check that the embeddings were extracted correctly
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result[0], np.array([0.1, 0.2, 0.3]))
        np.testing.assert_array_equal(result[1], np.array([0.4, 0.5, 0.6]))

        # Check that get_embeddings was called with the right arguments
        benchmarker.embeddings_provider.get_embeddings.assert_called_once_with(
            ["Q1", "Q2"]
        )

    @patch("qadst.embeddings.get_embeddings_model")
    def test_extract_embeddings_from_qa_pairs_no_model(self, mock_embeddings_model):
        """Test extracting embeddings without an embeddings model."""
        benchmarker = ClusterBenchmarker()

        # Try to extract embeddings without an embeddings model
        with pytest.raises(ValueError, match="Embeddings model not provided"):
            benchmarker.extract_embeddings_from_qa_pairs([("Q1", "A1"), ("Q2", "A2")])

    @patch("qadst.embeddings.get_embeddings_model")
    def test_extract_embeddings_from_qa_pairs_empty_list(self, mock_embeddings_model):
        """Test extracting embeddings from an empty list of QA pairs."""
        # Setup mock
        mock_embeddings_instance = MagicMock()
        mock_embeddings_model.return_value = mock_embeddings_instance

        # Create benchmarker with a mock embeddings model
        benchmarker = ClusterBenchmarker(embedding_model_name="test-model")

        # Create a mock embeddings provider
        benchmarker.embeddings_provider = MagicMock()

        # Mock the get_embeddings method
        benchmarker.embeddings_provider.get_embeddings = MagicMock(return_value=[])

        # Extract embeddings from an empty list of QA pairs
        result = benchmarker.extract_embeddings_from_qa_pairs([])

        # Check that the result is an empty array
        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)

        # Check that get_embeddings was called with an empty list
        benchmarker.embeddings_provider.get_embeddings.assert_called_once_with([])

    @patch("qadst.embeddings.get_embeddings_model")
    def test_extract_embeddings_from_qa_pairs_model_error(self, mock_embeddings_model):
        """Test extracting embeddings with a model error."""
        # Setup mock
        mock_embeddings_instance = MagicMock()
        mock_embeddings_model.return_value = mock_embeddings_instance

        # Create benchmarker with a mock embeddings model
        benchmarker = ClusterBenchmarker(embedding_model_name="test-model")

        # Create a mock embeddings provider
        benchmarker.embeddings_provider = MagicMock()

        # Mock the get_embeddings method to raise an exception
        benchmarker.embeddings_provider.get_embeddings = MagicMock(
            side_effect=Exception("Model error")
        )

        # Try to extract embeddings with a model error
        with pytest.raises(Exception, match="Model error"):
            benchmarker.extract_embeddings_from_qa_pairs([("Q1", "A1"), ("Q2", "A2")])
