"""Unit tests for the deterministic hash function."""

import hashlib
import tempfile
from unittest.mock import patch

from qadst import MockClusterer


def test_deterministic_hash_consistency():
    """Test that the hash function produces consistent results."""
    with patch("qadst.base.OpenAIEmbeddings"), patch("qadst.base.ChatOpenAI"):
        clusterer = MockClusterer(
            embedding_model_name="test-model",
            output_dir=tempfile.mkdtemp(),
        )

        # Test with a list of strings
        items = ["apple", "banana", "cherry"]

        # Calculate hash multiple times
        hash1 = clusterer._calculate_deterministic_hash(items)
        hash2 = clusterer._calculate_deterministic_hash(items)
        hash3 = clusterer._calculate_deterministic_hash(items)

        # All hashes should be identical
        assert hash1 == hash2 == hash3

        # Verify the hash is correct
        expected = hashlib.sha256("".join(sorted(items)).encode("utf-8")).hexdigest()
        assert hash1 == expected


def test_deterministic_hash_order_independence():
    """Test that the hash function is order-independent."""
    with patch("qadst.base.OpenAIEmbeddings"), patch("qadst.base.ChatOpenAI"):
        clusterer = MockClusterer(
            embedding_model_name="test-model",
            output_dir=tempfile.mkdtemp(),
        )

        # Test with different orderings of the same items
        items1 = ["apple", "banana", "cherry"]
        items2 = ["banana", "cherry", "apple"]
        items3 = ["cherry", "apple", "banana"]

        # Calculate hashes
        hash1 = clusterer._calculate_deterministic_hash(items1)
        hash2 = clusterer._calculate_deterministic_hash(items2)
        hash3 = clusterer._calculate_deterministic_hash(items3)

        # All hashes should be identical
        assert hash1 == hash2 == hash3


def test_deterministic_hash_different_inputs():
    """Test that the hash function produces different results for different inputs."""
    with patch("qadst.base.OpenAIEmbeddings"), patch("qadst.base.ChatOpenAI"):
        clusterer = MockClusterer(
            embedding_model_name="test-model",
            output_dir=tempfile.mkdtemp(),
        )

        # Test with different inputs
        hash1 = clusterer._calculate_deterministic_hash(["apple", "banana"])
        hash2 = clusterer._calculate_deterministic_hash(["apple", "cherry"])
        hash3 = clusterer._calculate_deterministic_hash(["banana", "cherry"])

        # All hashes should be different
        assert hash1 != hash2
        assert hash1 != hash3
        assert hash2 != hash3
