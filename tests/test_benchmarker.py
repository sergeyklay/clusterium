"""Tests for the benchmarker module."""

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
