"""Test utilities for the QA dataset clustering toolkit.

This module contains utilities that are primarily intended for testing
the toolkit. These utilities may also be useful for users who want to
write their own tests against the toolkit.
"""

from typing import Any, Dict, List, Tuple

from .base import BaseClusterer


class FakeClusterer(BaseClusterer):
    """Concrete implementation of BaseClusterer for testing.

    This class provides a minimal implementation of the abstract methods
    in BaseClusterer, making it suitable for unit testing the base class
    functionality without needing to use a full implementation like
    HDBSCANQAClusterer.

    Example:
        >>> from qadst.testing import FakeClusterer
        >>> clusterer = FakeClusterer(embedding_model_name="test-model")
        >>> clusterer.cluster_method()
        'test'
    """

    def cluster_questions(self, qa_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Implement abstract method with minimal functionality."""
        return {"clusters": []}

    def cluster_method(self) -> str:
        """Implement abstract method with a test identifier."""
        return "test"
