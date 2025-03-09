"""Common test fixtures, mocks and configurations."""

import os
import tempfile
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qadst import HDBSCANQAClusterer, MockClusterer


@pytest.fixture
def sample_qa_pairs() -> List[Tuple[str, str]]:
    """Return a sample list of QA pairs for testing."""
    return [
        ("How do I reset my password?", "Click the 'Forgot Password' link."),
        ("How can I change my password?", "Use the 'Forgot Password' option."),
        ("What payment methods do you accept?", "We accept credit cards and PayPal."),
        ("Can I pay with Bitcoin?", "Yes, we accept cryptocurrency payments."),
        ("How do I contact support?", "Email us at support@example.com."),
    ]


@pytest.fixture
def mock_embeddings() -> List[np.ndarray]:
    """Return mock embeddings for testing."""
    # Create deterministic embeddings where the first two are similar
    # and the third and fourth are similar
    return [
        np.array([0.9, 0.1, 0.1]),  # Password reset
        np.array([0.85, 0.15, 0.1]),  # Password change (similar to first)
        np.array([0.1, 0.9, 0.1]),  # Payment methods
        np.array([0.15, 0.85, 0.1]),  # Bitcoin (similar to payment)
        np.array([0.1, 0.1, 0.9]),  # Support contact (unique)
    ]


@pytest.fixture
def mock_base_clusterer():
    """Return a mock BaseClusterer for testing."""
    with patch("qadst.base.OpenAIEmbeddings"), patch("qadst.base.ChatOpenAI"):
        clusterer = MockClusterer(
            embedding_model_name="test-model",
            output_dir=tempfile.mkdtemp(),
        )

        # Mock the embeddings_model.embed_documents method
        clusterer.embeddings_model = MagicMock()
        clusterer.embeddings_model.embed_documents.return_value = [
            [0.9, 0.1, 0.1],  # Password reset
            [0.85, 0.15, 0.1],  # Password change (similar to first)
            [0.1, 0.9, 0.1],  # Payment methods
            [0.15, 0.85, 0.1],  # Bitcoin (similar to payment)
            [0.1, 0.1, 0.9],  # Support contact (unique)
        ]

        yield clusterer


@pytest.fixture
def mock_hdbscan_clusterer():
    """Return a mock HDBSCANQAClusterer for testing."""
    with (
        patch("qadst.clusterer.HDBSCAN"),
        patch("qadst.base.OpenAIEmbeddings"),
        patch("qadst.base.ChatOpenAI"),
    ):
        clusterer = HDBSCANQAClusterer(
            embedding_model_name="test-model",
            output_dir=tempfile.mkdtemp(),
        )

        # Mock the embeddings_model.embed_documents method
        clusterer.embeddings_model = MagicMock()
        clusterer.embeddings_model.embed_documents.return_value = [
            [0.9, 0.1, 0.1],  # Password reset
            [0.85, 0.15, 0.1],  # Password change (similar to first)
            [0.1, 0.9, 0.1],  # Payment methods
            [0.15, 0.85, 0.1],  # Bitcoin (similar to payment)
            [0.1, 0.1, 0.9],  # Support contact (unique)
        ]

        yield clusterer


@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file with sample QA pairs."""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as f:
        f.write("question,answer\n")
        f.write("How do I reset my password?,Click the 'Forgot Password' link.\n")
        f.write(
            "What payment methods do you accept?,We accept credit cards and PayPal.\n"
        )
        f.write("How do I contact support?,Email us at support@example.com.\n")
        temp_file_name = f.name

    yield temp_file_name

    # Clean up
    if os.path.exists(temp_file_name):
        os.unlink(temp_file_name)
