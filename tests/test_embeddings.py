"""Tests for the embeddings module."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np

from qadst.embeddings import EmbeddingsCache, EmbeddingsProvider, get_embeddings_model


class TestEmbeddingsCache:
    """Tests for the EmbeddingsCache class."""

    def test_init(self):
        """Test initialization of the cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingsCache(output_dir=temp_dir)
            assert cache.memory_cache == {}
            assert cache.output_dir == temp_dir
            assert cache.cache_dir == os.path.join(temp_dir, "embedding_cache")
            assert os.path.exists(cache.cache_dir)

    def test_get_not_in_cache(self):
        """Test getting embeddings that are not in the cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingsCache(output_dir=temp_dir)
            result = cache.get("test_key")
            assert result is None

    def test_set_and_get(self):
        """Test setting and getting embeddings from the cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingsCache(output_dir=temp_dir)
            embeddings = [np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])]
            cache.set("test_key", embeddings)

            # Check memory cache
            assert "test_key" in cache.memory_cache
            np.testing.assert_array_equal(
                cache.memory_cache["test_key"][0], embeddings[0]
            )
            np.testing.assert_array_equal(
                cache.memory_cache["test_key"][1], embeddings[1]
            )

            # Check disk cache
            cache_file = os.path.join(cache.cache_dir, "test_key.npy")
            assert os.path.exists(cache_file)

            # Get from cache
            result = cache.get("test_key")
            assert result is not None
            np.testing.assert_array_equal(result[0], embeddings[0])
            np.testing.assert_array_equal(result[1], embeddings[1])


class TestEmbeddingsProvider:
    """Tests for the EmbeddingsProvider class."""

    def test_init(self):
        """Test initialization of the provider."""
        model = MagicMock()
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = EmbeddingsProvider(model=model, output_dir=temp_dir)
            assert provider.model == model
            assert isinstance(provider.cache, EmbeddingsCache)
            assert provider.cache.output_dir == temp_dir

    def test_get_embeddings_no_cache(self):
        """Test getting embeddings without caching."""
        model = MagicMock()
        model.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        with tempfile.TemporaryDirectory() as temp_dir:
            provider = EmbeddingsProvider(model=model, output_dir=temp_dir)
            result = provider.get_embeddings(["text1", "text2"])

            model.embed_documents.assert_called_once_with(["text1", "text2"])
            assert len(result) == 2
            np.testing.assert_array_equal(result[0], np.array([0.1, 0.2, 0.3]))
            np.testing.assert_array_equal(result[1], np.array([0.4, 0.5, 0.6]))

    def test_get_embeddings_with_cache(self):
        """Test getting embeddings with caching."""
        model = MagicMock()
        model.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        with tempfile.TemporaryDirectory() as temp_dir:
            provider = EmbeddingsProvider(model=model, output_dir=temp_dir)

            # First call should compute embeddings
            result1 = provider.get_embeddings(["text1", "text2"], cache_key="test_key")
            model.embed_documents.assert_called_once_with(["text1", "text2"])

            # Reset mock to check if it's called again
            model.embed_documents.reset_mock()

            # Second call should use cache
            result2 = provider.get_embeddings(["text1", "text2"], cache_key="test_key")
            model.embed_documents.assert_not_called()

            # Results should be the same
            np.testing.assert_array_equal(result1[0], result2[0])
            np.testing.assert_array_equal(result1[1], result2[1])

    def test_calculate_cosine_similarity(self):
        """Test calculating cosine similarity."""
        model = MagicMock()
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = EmbeddingsProvider(model=model, output_dir=temp_dir)

            # Test with orthogonal vectors
            vec1 = np.array([1.0, 0.0, 0.0])
            vec2 = np.array([0.0, 1.0, 0.0])
            similarity = provider.calculate_cosine_similarity(vec1, vec2)
            assert similarity == 0.0

            # Test with identical vectors
            vec1 = np.array([0.1, 0.2, 0.3])
            vec2 = np.array([0.1, 0.2, 0.3])
            similarity = provider.calculate_cosine_similarity(vec1, vec2)
            assert similarity == 1.0

            # Test with similar vectors
            vec1 = np.array([0.1, 0.2, 0.3])
            vec2 = np.array([0.2, 0.3, 0.4])
            similarity = provider.calculate_cosine_similarity(vec1, vec2)
            assert 0.9 < similarity < 1.0


@patch("qadst.embeddings.OpenAIEmbeddings")
def test_get_embeddings_model(mock_openai_embeddings):
    """Test the get_embeddings_model factory function."""
    # Setup mock
    mock_instance = MagicMock()
    mock_openai_embeddings.return_value = mock_instance

    # Call the function
    model = get_embeddings_model("text-embedding-3-large", api_key="test_key")

    # Check that OpenAIEmbeddings was called with the right arguments
    mock_openai_embeddings.assert_called_once()
    call_args = mock_openai_embeddings.call_args[1]
    assert call_args["model"] == "text-embedding-3-large"
    assert call_args["api_key"] is not None

    # Check that the returned model is the mock instance
    assert model == mock_instance
