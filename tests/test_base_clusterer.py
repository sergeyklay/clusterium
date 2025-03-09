"""Unit tests for the BaseClusterer class."""

import hashlib
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

from qadst import MockClusterer


def test_calculate_cosine_similarity(mock_base_clusterer):
    """Test the calculate_cosine_similarity method."""
    # Test with orthogonal vectors (should be 0)
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0, 1, 0])
    similarity = mock_base_clusterer.calculate_cosine_similarity(vec1, vec2)
    assert np.isclose(similarity, 0.0)

    # Test with identical vectors (should be 1)
    vec1 = np.array([0.5, 0.5, 0.5])
    vec2 = np.array([0.5, 0.5, 0.5])
    similarity = mock_base_clusterer.calculate_cosine_similarity(vec1, vec2)
    assert np.isclose(similarity, 1.0)

    # Test with similar vectors
    vec1 = np.array([0.9, 0.1, 0.0])
    vec2 = np.array([0.8, 0.2, 0.0])
    similarity = mock_base_clusterer.calculate_cosine_similarity(vec1, vec2)
    assert 0.9 < similarity < 1.0


def test_calculate_deterministic_hash(mock_base_clusterer):
    """Test the _calculate_deterministic_hash method."""
    # Test with a single item
    items = ["test"]
    hash_value = mock_base_clusterer._calculate_deterministic_hash(items)
    expected = hashlib.sha256("test".encode("utf-8")).hexdigest()
    assert hash_value == expected

    # Test with multiple items (should be sorted)
    items = ["b", "a", "c"]
    hash_value = mock_base_clusterer._calculate_deterministic_hash(items)
    expected = hashlib.sha256("abc".encode("utf-8")).hexdigest()
    assert hash_value == expected

    # Test with empty list
    items = []
    hash_value = mock_base_clusterer._calculate_deterministic_hash(items)
    expected = hashlib.sha256("".encode("utf-8")).hexdigest()
    assert hash_value == expected


def test_load_qa_pairs(temp_csv_file):
    """Test the load_qa_pairs method."""
    with patch("qadst.base.OpenAIEmbeddings"), patch("qadst.base.ChatOpenAI"):
        clusterer = MockClusterer(
            embedding_model_name="test-model",
            output_dir=tempfile.mkdtemp(),
        )

        # Test loading from a valid CSV file
        qa_pairs = clusterer.load_qa_pairs(temp_csv_file)
        assert len(qa_pairs) == 3
        assert qa_pairs[0] == (
            "How do I reset my password?",
            "Click the 'Forgot Password' link.",
        )

        # Test with a non-existent file
        with pytest.raises(FileNotFoundError):
            clusterer.load_qa_pairs("non_existent_file.csv")

        # Test with an invalid CSV file (create a temporary file with wrong format)
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as f:
            f.write("invalid,format,headers\n")
            f.write("some,data,here\n")
            invalid_file = f.name

        try:
            with pytest.raises(ValueError):
                clusterer.load_qa_pairs(invalid_file)
        finally:
            if os.path.exists(invalid_file):
                os.unlink(invalid_file)
