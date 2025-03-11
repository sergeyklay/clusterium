"""Tests for the DirichletProcess and PitmanYorProcess classes using fixtures."""

import os
from unittest.mock import patch

from clusx.clustering.cache import EmbeddingCache
from clusx.clustering.models import DirichletProcess, PitmanYorProcess


class TestDirichletProcessWithFixtures:
    """Tests for DirichletProcess using fixtures."""

    @patch("clusx.clustering.models.SentenceTransformer")
    def test_fit_with_sample_texts(self, mock_st, sample_texts, tmp_path):
        """Test fitting the model with sample texts from fixtures."""
        # Create a cache directory
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create a cache provider
        cache = EmbeddingCache(cache_dir=str(cache_dir))

        # Configure the mock to return a fixed embedding
        mock_instance = mock_st.return_value
        mock_instance.encode.return_value = [0.1, 0.2, 0.3, 0.4]

        # Create a DirichletProcess instance
        dp = DirichletProcess(alpha=1.0, cache=cache)

        # Fit the model
        clusters, _ = dp.fit(sample_texts)

        # Check that we have the correct number of cluster assignments
        assert len(clusters) == len(sample_texts)

        # Check that the cache file was created
        cache_file = os.path.join(str(cache_dir), "embeddings.pkl")
        assert os.path.exists(cache_file)


class TestPitmanYorProcessWithFixtures:
    """Tests for PitmanYorProcess using fixtures."""

    @patch("clusx.clustering.models.SentenceTransformer")
    def test_fit_with_sample_texts(self, mock_st, sample_texts, tmp_path):
        """Test fitting the model with sample texts from fixtures."""
        # Create a cache directory
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create a cache provider
        cache = EmbeddingCache(cache_dir=str(cache_dir))

        # Configure the mock to return a fixed embedding
        mock_instance = mock_st.return_value
        mock_instance.encode.return_value = [0.1, 0.2, 0.3, 0.4]

        # Create a PitmanYorProcess instance
        pyp = PitmanYorProcess(alpha=1.0, sigma=0.5, cache=cache)

        # Fit the model
        clusters, _ = pyp.fit(sample_texts)

        # Check that we have the correct number of cluster assignments
        assert len(clusters) == len(sample_texts)

        # Check that the cache file was created
        cache_file = os.path.join(str(cache_dir), "embeddings.pkl")
        assert os.path.exists(cache_file)

    @patch("clusx.clustering.models.SentenceTransformer")
    def test_cluster_sizes_tracking(self, mock_st, sample_texts, tmp_path):
        """Test that cluster parameters are properly tracked in PitmanYorProcess."""
        # Create a cache directory
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create a cache provider
        cache = EmbeddingCache(cache_dir=str(cache_dir))

        # Configure the mock to return a fixed embedding
        mock_instance = mock_st.return_value
        mock_instance.encode.return_value = [0.1, 0.2, 0.3, 0.4]

        # Create a PitmanYorProcess instance
        pyp = PitmanYorProcess(alpha=1.0, sigma=0.5, cache=cache)

        # Fit the model
        clusters, _ = pyp.fit(sample_texts)

        # Check that cluster_params has entries for all unique clusters
        unique_clusters = set(clusters)
        for cluster_id in unique_clusters:
            assert cluster_id in pyp.cluster_params
