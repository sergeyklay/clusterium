"""Common test fixtures, mocks and configurations."""

import os
import tempfile
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qadst import FakeClusterer, HDBSCANQAClusterer


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
def filter_qa_pairs() -> List[Tuple[str, str]]:
    """Return a sample list of QA pairs for testing filtering."""
    return [
        # Client questions
        ("How do I reset my password?", "Click the 'Forgot Password' link."),
        ("What payment methods do you accept?", "We accept credit cards and PayPal."),
        ("How do I contact support?", "Email us at support@example.com."),
        # Engineering questions
        (
            "What's the expected API latency in EU region?",
            "Under 100ms with proper connection pooling.",
        ),
        (
            "How is the database sharded?",
            "We use hash-based sharding on the customer_id field.",
        ),
    ]


@pytest.fixture
def mock_embeddings() -> List[np.ndarray]:
    """Return mock embeddings for testing."""
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
    with patch("qadst.embeddings.get_embeddings_model"), patch("qadst.base.ChatOpenAI"):
        mock_embeddings_provider = MagicMock()
        mock_embeddings_provider.get_model_name.return_value = "test-model"

        def mock_cosine_similarity(vec1, vec2):
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1_np, vec2_np) / (norm1 * norm2))

        mock_embeddings_provider.calculate_cosine_similarity.side_effect = (
            mock_cosine_similarity
        )

        clusterer = FakeClusterer(
            embeddings_provider=mock_embeddings_provider,
            output_dir=tempfile.mkdtemp(),
        )

        yield clusterer


@pytest.fixture
def mock_filter_clusterer():
    """Return a mock BaseClusterer with LLM for testing filtering."""
    with (
        patch("qadst.base.OpenAIEmbeddings"),
        patch("qadst.base.ChatOpenAI"),
        patch("qadst.base.PromptTemplate"),
        patch("qadst.base.os.path.exists", return_value=True),
    ):
        output_dir = tempfile.mkdtemp()

        mock_embeddings_provider = MagicMock()
        mock_embeddings_provider.get_model_name.return_value = "test-model"

        clusterer = FakeClusterer(
            embeddings_provider=mock_embeddings_provider,
            llm_model_name="test-llm",
            output_dir=output_dir,
        )

        clusterer.filter_cache = {}
        clusterer.llm = MagicMock()

        mock_response = MagicMock()
        mock_response.content = "[false, false, false, true, true]"
        clusterer.llm.invoke.return_value = mock_response

        original_classify = clusterer._classify_questions_batch

        def mock_classify_batch(questions):
            if len(questions) == 5:
                return [False, False, False, True, True]
            elif len(questions) == 3:  # For the cache test
                return [False, True, True]
            elif len(questions) == 2:  # For batch processing test (first batch)
                return [False, False]
            else:
                return original_classify(questions)

        clusterer._classify_questions_batch = mock_classify_batch

        yield clusterer

        if os.path.exists(output_dir):
            import shutil

            shutil.rmtree(output_dir)


@pytest.fixture
def mock_hdbscan_clusterer():
    """Return a mock HDBSCANQAClusterer for testing."""
    with (
        patch("qadst.clusterer.HDBSCAN"),
        patch("qadst.embeddings.get_embeddings_model"),
        patch("qadst.base.ChatOpenAI"),
    ):
        mock_embeddings_provider = MagicMock()
        mock_embeddings_provider.get_model_name.return_value = "test-model"

        def mock_cosine_similarity(vec1, vec2):
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1_np, vec2_np) / (norm1 * norm2))

        mock_embeddings_provider.calculate_cosine_similarity.side_effect = (
            mock_cosine_similarity
        )

        mock_embeddings_provider.get_embeddings.return_value = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
            np.array([0.7, 0.8, 0.9]),
        ]

        clusterer = HDBSCANQAClusterer(
            embeddings_provider=mock_embeddings_provider,
            output_dir=tempfile.mkdtemp(),
        )

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

    if os.path.exists(temp_file_name):
        os.unlink(temp_file_name)
