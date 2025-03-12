"""Tests for the DirichletProcess and PitmanYorProcess classes using fixtures."""

from unittest.mock import patch

from clusx.clustering.models import DirichletProcess, PitmanYorProcess


class TestDirichletProcessWithFixtures:
    """Tests for DirichletProcess using fixtures."""

    @patch("clusx.clustering.models.SentenceTransformer")
    def test_fit_with_sample_texts(self, mock_st, sample_texts):
        """Test fitting the model with sample texts from fixtures."""
        # Configure the mock to return a fixed embedding
        mock_instance = mock_st.return_value
        mock_instance.encode.return_value = [0.1, 0.2, 0.3, 0.4]

        # Create a DirichletProcess instance
        dp = DirichletProcess(alpha=1.0)

        # Fit the model
        clusters, _ = dp.fit(sample_texts)

        # Check that we have the correct number of cluster assignments
        assert len(clusters) == len(sample_texts)


class TestPitmanYorProcessWithFixtures:
    """Tests for PitmanYorProcess using fixtures."""

    @patch("clusx.clustering.models.SentenceTransformer")
    def test_fit_with_sample_texts(self, mock_st, sample_texts):
        """Test fitting the model with sample texts from fixtures."""
        # Configure the mock to return a fixed embedding
        mock_instance = mock_st.return_value
        mock_instance.encode.return_value = [0.1, 0.2, 0.3, 0.4]

        # Create a PitmanYorProcess instance
        pyp = PitmanYorProcess(alpha=1.0, sigma=0.5)

        # Fit the model
        clusters, _ = pyp.fit(sample_texts)

        # Check that we have the correct number of cluster assignments
        assert len(clusters) == len(sample_texts)

    @patch("clusx.clustering.models.SentenceTransformer")
    def test_cluster_sizes_tracking(self, mock_st, sample_texts):
        """Test that cluster parameters are properly tracked in PitmanYorProcess."""
        # Configure the mock to return a fixed embedding
        mock_instance = mock_st.return_value
        mock_instance.encode.return_value = [0.1, 0.2, 0.3, 0.4]

        # Create a PitmanYorProcess instance
        pyp = PitmanYorProcess(alpha=1.0, sigma=0.5)

        # Fit the model
        clusters, _ = pyp.fit(sample_texts)

        # Check that cluster_params has entries for all unique clusters
        unique_clusters = set(clusters)
        for cluster_id in unique_clusters:
            assert cluster_id in pyp.cluster_params
