"""Unit tests for the EmbeddingCache class."""

import os
import pickle
from unittest.mock import mock_open, patch

import pytest
import torch

from clusx.clustering.cache import EmbeddingCache


@pytest.fixture
def embedding_cache(tmp_path):
    """Create an EmbeddingCache instance with a temporary directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return EmbeddingCache(cache_dir=str(cache_dir))


@pytest.fixture
def sample_embedding():
    """Create a sample tensor embedding."""
    return torch.tensor([0.1, 0.2, 0.3])


def test_init_with_cache_dir(tmp_path):
    """Test initializing EmbeddingCache with a cache directory."""
    cache_dir = str(tmp_path / "cache")
    cache = EmbeddingCache(cache_dir=cache_dir)

    assert cache.cache_dir == cache_dir
    assert cache.embedding_cache == {}


def test_init_without_cache_dir():
    """Test initializing EmbeddingCache without a cache directory."""
    cache = EmbeddingCache()

    assert cache.cache_dir is None
    assert cache.embedding_cache == {}


def test_get_cache_path(embedding_cache):
    """Test get_cache_path method."""
    base_name = "test_embeddings"
    expected_path = os.path.join(embedding_cache.cache_dir, f"{base_name}.pkl")

    assert embedding_cache.get_cache_path(base_name) == expected_path


def test_get_cache_path_with_special_chars(embedding_cache):
    """Test get_cache_path method with special characters in base_name."""
    base_name = "test/embeddings.with spaces"
    expected_path = os.path.join(
        embedding_cache.cache_dir, "test_embeddings_with_spaces.pkl"
    )

    assert embedding_cache.get_cache_path(base_name) == expected_path


def test_get_cache_path_without_cache_dir():
    """Test get_cache_path method when cache_dir is None."""
    cache = EmbeddingCache()

    assert cache.get_cache_path("test") is None


def test_hash_key(embedding_cache):
    """Test _hash_key method."""
    key = "test_key"
    hashed_key1 = embedding_cache._hash_key(key)
    hashed_key2 = embedding_cache._hash_key(key)

    # Same key should produce same hash
    assert hashed_key1 == hashed_key2

    # Different keys should produce different hashes
    assert embedding_cache._hash_key("different_key") != hashed_key1


def test_set_and_get(embedding_cache, sample_embedding):
    """Test set and get methods."""
    key = "test_key"

    # Initially, key should not be in cache
    assert embedding_cache.get(key) is None

    # Set the embedding
    embedding_cache.set(key, sample_embedding)

    # Now the key should be in cache
    retrieved = embedding_cache.get(key)
    assert retrieved is not None
    assert torch.equal(retrieved, sample_embedding)


def test_contains(embedding_cache, sample_embedding):
    """Test __contains__ method."""
    key = "test_key"

    # Initially, key should not be in cache
    assert key not in embedding_cache

    # Set the embedding
    embedding_cache.set(key, sample_embedding)

    # Now the key should be in cache
    assert key in embedding_cache


def test_load_cache_no_cache_dir():
    """Test load_cache method when cache_dir is None."""
    cache = EmbeddingCache()

    assert cache.load_cache() == {}


def test_load_cache_file_not_exists(embedding_cache):
    """Test load_cache method when cache file doesn't exist."""
    assert embedding_cache.load_cache() == {}


def test_load_cache_file_exists(embedding_cache, sample_embedding):
    """Test load_cache method when cache file exists."""
    # Create a cache file
    cache_path = embedding_cache.get_cache_path("embeddings")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # Create a sample cache dictionary
    key = "test_key"
    hashed_key = embedding_cache._hash_key(key)
    cache_data = {hashed_key: sample_embedding}

    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)

    # Load the cache
    loaded_cache = embedding_cache.load_cache()

    # Verify the loaded cache
    assert hashed_key in loaded_cache
    assert torch.equal(loaded_cache[hashed_key], sample_embedding)


def test_load_cache_with_exception(embedding_cache):
    """Test load_cache method when an exception occurs."""
    with patch("builtins.open", mock_open()) as mock_file:
        mock_file.side_effect = Exception("Test exception")

        # Load the cache, which should handle the exception
        loaded_cache = embedding_cache.load_cache()

        # Verify the loaded cache is empty
        assert loaded_cache == {}


def test_save_cache_no_cache_dir():
    """Test save_cache method when cache_dir is None."""
    cache = EmbeddingCache()

    # This should not raise an exception
    cache.save_cache()


def test_save_cache_empty_cache(embedding_cache):
    """Test save_cache method when embedding_cache is empty."""
    # This should not create a file
    embedding_cache.save_cache()

    cache_path = embedding_cache.get_cache_path("embeddings")
    assert not os.path.exists(cache_path)


def test_save_cache(embedding_cache, sample_embedding):
    """Test save_cache method."""
    key = "test_key"
    embedding_cache.set(key, sample_embedding)

    # Save the cache
    embedding_cache.save_cache()

    # Verify the cache file was created
    cache_path = embedding_cache.get_cache_path("embeddings")
    assert os.path.exists(cache_path)

    # Load the cache file and verify its contents
    with open(cache_path, "rb") as f:
        loaded_data = pickle.load(f)

    hashed_key = embedding_cache._hash_key(key)
    assert hashed_key in loaded_data
    assert torch.equal(loaded_data[hashed_key], sample_embedding)


def test_save_cache_custom_base_name(embedding_cache, sample_embedding):
    """Test save_cache method with a custom base_name."""
    key = "test_key"
    embedding_cache.set(key, sample_embedding)

    # Save the cache with a custom base_name
    custom_base_name = "custom_embeddings"
    embedding_cache.save_cache(custom_base_name)

    # Verify the cache file was created with the custom base_name
    cache_path = embedding_cache.get_cache_path(custom_base_name)
    assert os.path.exists(cache_path)

    # Load the cache file and verify its contents
    with open(cache_path, "rb") as f:
        loaded_data = pickle.load(f)

    hashed_key = embedding_cache._hash_key(key)
    assert hashed_key in loaded_data
    assert torch.equal(loaded_data[hashed_key], sample_embedding)
