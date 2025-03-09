"""Tests for the benchmarker module."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

from qadst.benchmarker import ClusterBenchmarker
from qadst.reporters import ConsoleReporter, CSVReporter


class TestClusterBenchmarker:
    """Tests for the ClusterBenchmarker class."""

    @patch("qadst.benchmarker.OpenAIEmbeddings")
    @patch("qadst.benchmarker.ChatOpenAI")
    @patch("qadst.benchmarker.ReporterRegistry")
    def test_init_default_values(
        self, mock_registry, mock_chat_openai, mock_embeddings
    ):
        """Test initialization with default values."""
        # Setup mock registry
        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        # Create benchmarker with default values
        benchmarker = ClusterBenchmarker()

        # Check default values
        assert benchmarker.output_dir == "./output"
        assert benchmarker.embeddings_model is None
        assert benchmarker.llm is None

        # Check that the reporter registry was initialized
        assert mock_registry.called
        assert benchmarker.reporter_registry == mock_registry_instance

        # Check that the reporters were registered
        assert mock_registry_instance.register.call_count == 2

        # Verify the first call was to register the CSV reporter
        args, kwargs = mock_registry_instance.register.call_args_list[0]
        assert args[0] == "csv"
        assert isinstance(args[1], CSVReporter)
        assert kwargs.get("enabled", False) is True

        # Verify the second call was to register the console reporter
        args, kwargs = mock_registry_instance.register.call_args_list[1]
        assert args[0] == "console"
        assert isinstance(args[1], ConsoleReporter)
        assert kwargs.get("enabled", False) is True

    @patch("qadst.benchmarker.OpenAIEmbeddings")
    @patch("qadst.benchmarker.ChatOpenAI")
    def test_init_with_models(self, mock_chat_openai, mock_embeddings):
        """Test initialization with model names."""
        # Setup mocks
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance

        mock_llm_instance = MagicMock()
        mock_chat_openai.return_value = mock_llm_instance

        # Create temp directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create benchmarker with custom values
            benchmarker = ClusterBenchmarker(
                embedding_model_name="text-embedding-3-large",
                llm_model_name="gpt-4o",
                output_dir=temp_dir,
            )

            # Check values
            assert benchmarker.output_dir == temp_dir
            assert benchmarker.embeddings_model == mock_embeddings_instance
            assert benchmarker.llm == mock_llm_instance

            # Verify the embedding model was initialized correctly
            mock_embeddings.assert_called_once_with(model="text-embedding-3-large")

            # Verify the LLM was initialized correctly
            mock_chat_openai.assert_called_once_with(model="gpt-4o", temperature=0.0)

            # Verify the output directory was created
            assert os.path.exists(temp_dir)

    @patch("qadst.benchmarker.OpenAIEmbeddings")
    @patch("qadst.benchmarker.ChatOpenAI")
    @patch("qadst.benchmarker.logger")
    def test_init_with_model_errors(
        self, mock_logger, mock_chat_openai, mock_embeddings
    ):
        """Test initialization with model errors."""
        # Setup mocks to raise exceptions
        mock_embeddings.side_effect = Exception("Embedding model error")
        mock_chat_openai.side_effect = Exception("LLM error")

        # Create benchmarker with models that will fail
        benchmarker = ClusterBenchmarker(
            embedding_model_name="invalid-model", llm_model_name="invalid-llm"
        )

        # Check that models are None due to errors
        assert benchmarker.embeddings_model is None
        assert benchmarker.llm is None

        # Verify warning logs were created
        assert mock_logger.warning.call_count == 2
        mock_logger.warning.assert_any_call(
            "Failed to initialize embeddings model: Embedding model error"
        )
        mock_logger.warning.assert_any_call("Failed to initialize LLM: LLM error")

    @patch("qadst.benchmarker.OpenAIEmbeddings")
    @patch("qadst.benchmarker.ChatOpenAI")
    def test_load_clusters_valid_json(self, mock_chat_openai, mock_embeddings):
        """Test loading clusters from a valid JSON file."""
        # Create test data
        test_clusters = {
            "clusters": [
                {
                    "id": 1,
                    "representative": [{"question": "Test Q1", "answer": "Test A1"}],
                    "source": [
                        {"question": "Test Q1", "answer": "Test A1"},
                        {"question": "Test Q2", "answer": "Test A2"},
                    ],
                },
                {
                    "id": 2,
                    "representative": [{"question": "Test Q3", "answer": "Test A3"}],
                    "source": [{"question": "Test Q3", "answer": "Test A3"}],
                },
            ]
        }

        # Create a temporary file with test data
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            json.dump(test_clusters, f)
            temp_file_name = f.name

        try:
            # Create benchmarker
            benchmarker = ClusterBenchmarker()

            # Load clusters
            loaded_clusters = benchmarker.load_clusters(temp_file_name)

            # Verify the loaded data
            assert loaded_clusters == test_clusters
            assert len(loaded_clusters["clusters"]) == 2
            assert loaded_clusters["clusters"][0]["id"] == 1
            assert loaded_clusters["clusters"][1]["id"] == 2
            assert len(loaded_clusters["clusters"][0]["source"]) == 2
            assert len(loaded_clusters["clusters"][1]["source"]) == 1

        finally:
            # Clean up
            if os.path.exists(temp_file_name):
                os.unlink(temp_file_name)

    @patch("qadst.benchmarker.OpenAIEmbeddings")
    @patch("qadst.benchmarker.ChatOpenAI")
    def test_load_clusters_invalid_json(self, mock_chat_openai, mock_embeddings):
        """Test loading clusters from an invalid JSON file."""
        # Create a temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
            f.write("This is not valid JSON")
            temp_file_name = f.name

        try:
            # Create benchmarker
            benchmarker = ClusterBenchmarker()

            # Attempt to load clusters from invalid JSON
            with patch("qadst.benchmarker.json.load") as mock_json_load:
                mock_json_load.side_effect = json.JSONDecodeError("Test error", "", 0)

                # Should raise JSONDecodeError
                try:
                    benchmarker.load_clusters(temp_file_name)
                    assert False, "Expected JSONDecodeError was not raised"
                except json.JSONDecodeError:
                    pass  # Test passed

        finally:
            # Clean up
            if os.path.exists(temp_file_name):
                os.unlink(temp_file_name)

    @patch("qadst.benchmarker.OpenAIEmbeddings")
    @patch("qadst.benchmarker.ChatOpenAI")
    def test_load_clusters_file_not_found(self, mock_chat_openai, mock_embeddings):
        """Test loading clusters from a non-existent file."""
        # Create benchmarker
        benchmarker = ClusterBenchmarker()

        # Attempt to load clusters from non-existent file
        non_existent_file = "/path/to/non/existent/file.json"

        # Should raise FileNotFoundError
        try:
            benchmarker.load_clusters(non_existent_file)
            assert False, "Expected FileNotFoundError was not raised"
        except FileNotFoundError:
            pass  # Test passed
