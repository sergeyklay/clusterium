"""Integration tests for the DirichletProcess and PitmanYorProcess classes."""

import os
import tempfile

import pytest

from clusx.clustering.cache import EmbeddingCache
from clusx.clustering.models import DirichletProcess, PitmanYorProcess


@pytest.fixture
def sample_texts():
    """Return a list of sample texts for clustering."""
    return [
        "What is Python?",
        "How do I install Python?",
        "What is TensorFlow?",
        "How do I install TensorFlow?",
        "What is PyTorch?",
        "How do I install PyTorch?",
    ]


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestDirichletProcessIntegration:
    """Integration tests for DirichletProcess."""

    def test_fit_with_real_data(self, sample_texts, temp_cache_dir):
        """Test fitting the model with real data."""
        # Create a cache provider
        cache = EmbeddingCache(cache_dir=temp_cache_dir)

        # Create a DirichletProcess instance
        dp = DirichletProcess(alpha=1.0, cache=cache)

        # Fit the model
        clusters, _ = dp.fit(sample_texts)

        # Check that we have the correct number of cluster assignments
        assert len(clusters) == len(sample_texts)

        # Check that the cache file was created
        cache_file = os.path.join(temp_cache_dir, "embeddings.pkl")
        assert os.path.exists(cache_file)


class TestPitmanYorProcessIntegration:
    """Integration tests for PitmanYorProcess."""

    def test_fit_with_real_data(self, sample_texts, temp_cache_dir):
        """Test fitting the model with real data."""
        # Create a cache provider
        cache = EmbeddingCache(cache_dir=temp_cache_dir)

        # Create a PitmanYorProcess instance
        pyp = PitmanYorProcess(alpha=1.0, sigma=0.5, cache=cache)

        # Fit the model
        clusters, _ = pyp.fit(sample_texts)

        # Check that we have the correct number of cluster assignments
        assert len(clusters) == len(sample_texts)

        # Check that the cache file was created
        cache_file = os.path.join(temp_cache_dir, "embeddings.pkl")
        assert os.path.exists(cache_file)

    def test_compare_with_dirichlet_process(self, sample_texts, temp_cache_dir):
        """Compare PitmanYorProcess with DirichletProcess."""
        # Create cache providers
        dp_cache = EmbeddingCache(cache_dir=os.path.join(temp_cache_dir, "dp"))
        pyp_cache = EmbeddingCache(cache_dir=os.path.join(temp_cache_dir, "pyp"))

        # Create model instances
        dp = DirichletProcess(alpha=1.0, cache=dp_cache)
        pyp = PitmanYorProcess(alpha=1.0, sigma=0.5, cache=pyp_cache)

        # Fit both models
        dp_clusters, _ = dp.fit(sample_texts)
        pyp_clusters, _ = pyp.fit(sample_texts)

        # Both should have the same number of cluster assignments
        assert len(dp_clusters) == len(pyp_clusters)

        # The number of unique clusters might be different due to the
        # different clustering algorithms
        dp_unique_clusters = len(set(dp_clusters))
        pyp_unique_clusters = len(set(pyp_clusters))

        # Just verify that we have at least one cluster
        assert dp_unique_clusters > 0
        assert pyp_unique_clusters > 0
