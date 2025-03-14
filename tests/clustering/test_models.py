"""Unit tests for the DirichletProcess and PitmanYorProcess classes."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from clusx.clustering.models import DirichletProcess, PitmanYorProcess


@pytest.fixture
def mock_sentence_transformer():
    """Create a mock SentenceTransformer."""
    with patch("clusx.clustering.models.SentenceTransformer") as mock_st:
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        mock_st.return_value = mock_instance
        yield mock_st


@pytest.fixture
def sample_embedding():
    """Return a sample embedding tensor."""
    return torch.tensor([0.1, 0.2, 0.3, 0.4])


class TestDirichletProcess:
    """Tests for the DirichletProcess class."""

    def test_init(self):
        """Test initialization of DirichletProcess."""
        dp = DirichletProcess(alpha=1.0)

        assert dp.alpha == 1.0
        assert dp.clusters == []
        assert isinstance(dp.cluster_params, dict)
        assert dp.similarity_metric == dp.cosine_similarity

    @patch("clusx.clustering.models.SentenceTransformer")
    def test_get_embedding_new(self, mock_st):
        """Test getting a new embedding."""
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        mock_st.return_value = mock_instance

        dp = DirichletProcess(alpha=1.0)
        embedding = dp.get_embedding("test text")

        mock_instance.encode.assert_called_once_with(
            "test text", show_progress_bar=False
        )
        assert np.array_equal(embedding, np.array([0.1, 0.2, 0.3, 0.4]))

    def test_cosine_similarity(self):
        """Test the cosine_similarity method."""
        dp = DirichletProcess(alpha=1.0)

        text_embedding = torch.tensor([1.0, 0.0, 0.0, 0.0])
        cluster_embedding = torch.tensor([0.0, 1.0, 0.0, 0.0])
        similarity = dp.cosine_similarity(text_embedding, cluster_embedding)

        assert similarity == 0.0

    @patch("numpy.random.choice")
    def test_assign_cluster_new(self, mock_choice, mock_sentence_transformer):
        """Test assigning a text to a new cluster."""
        dp = DirichletProcess(alpha=1.0)

        sample_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        with patch.object(dp, "get_embedding", return_value=sample_embedding):
            cluster_id = dp.assign_cluster("test text")

            assert len(dp.clusters) == 1
            assert len(dp.cluster_params) == 1
            assert dp.clusters[0] == 0
            assert cluster_id == 0
            assert 0 in dp.cluster_params

    @patch("numpy.random.choice")
    def test_assign_cluster_existing(self, mock_choice, mock_sentence_transformer):
        """Test assigning a text to an existing cluster."""
        dp = DirichletProcess(alpha=1.0)

        sample_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        dp.cluster_params[0] = {"mean": sample_embedding, "count": 1}
        dp.clusters.append(0)

        dp.random_state = MagicMock()
        dp.random_state.choice.return_value = 0

        test_embedding = np.array([0.5, 0.6, 0.7, 0.8])
        with patch.object(dp, "get_embedding", return_value=test_embedding):
            cluster_id = dp.assign_cluster("test text")

            assert len(dp.clusters) == 2
            assert len(dp.cluster_params) == 1
            assert dp.clusters[0] == 0
            assert dp.clusters[1] == 0
            assert cluster_id == 0
            assert dp.cluster_params[0]["count"] == 2

    def test_fit(self):
        """Test the fit method."""
        dp = DirichletProcess(alpha=1.0)

        # Patch the assign_cluster method to avoid randomness
        with patch.object(dp, "assign_cluster") as mock_assign:
            texts = ["text1", "text2", "text3"]
            clusters, params = dp.fit(texts)

            assert mock_assign.call_count == 3
            assert clusters is dp.clusters
            assert params is dp.cluster_params


class TestPitmanYorProcess:
    """Tests for the PitmanYorProcess class."""

    def test_init(self):
        """Test initialization of PitmanYorProcess."""
        pyp = PitmanYorProcess(alpha=1.0, sigma=0.5)

        assert pyp.alpha == 1.0
        assert pyp.sigma == 0.5
        assert pyp.clusters == []
        assert isinstance(pyp.cluster_params, dict)
        assert pyp.similarity_metric == pyp.cosine_similarity

    @patch("numpy.random.choice")
    @patch("clusx.clustering.models.cosine", return_value=0.2)
    def test_assign_cluster_new(
        self, mock_cosine, mock_choice, mock_sentence_transformer
    ):
        """Test assigning a text to a new cluster in PitmanYorProcess."""
        pyp = PitmanYorProcess(alpha=1.0, sigma=0.5)

        sample_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        with patch.object(pyp, "get_embedding", return_value=sample_embedding):
            cluster_id = pyp.assign_cluster("test text")

            assert len(pyp.clusters) == 1
            assert len(pyp.cluster_params) == 1
            assert pyp.clusters[0] == 0
            assert cluster_id == 0
            assert 0 in pyp.cluster_params

    @patch("numpy.random.choice")
    @patch("clusx.clustering.models.cosine", return_value=0.2)
    def test_assign_cluster_existing(
        self, mock_cosine, mock_choice, mock_sentence_transformer
    ):
        """Test assigning a text to an existing cluster in PitmanYorProcess."""
        pyp = PitmanYorProcess(alpha=1.0, sigma=0.5)

        # Add a cluster parameter with numpy array
        sample_embedding = np.array([0.1, 0.2, 0.3, 0.4])
        pyp.cluster_params[0] = {"mean": sample_embedding, "count": 1}
        pyp.clusters.append(0)

        pyp.random_state = MagicMock()
        pyp.random_state.choice.return_value = 0

        test_embedding = np.array([0.5, 0.6, 0.7, 0.8])
        with patch.object(pyp, "get_embedding", return_value=test_embedding):
            cluster_id = pyp.assign_cluster("test text")

            assert len(pyp.clusters) == 2
            assert len(pyp.cluster_params) == 1
            assert pyp.clusters == [0, 0]
            assert cluster_id == 0
            assert pyp.cluster_params[0]["count"] == 2

    def test_fit(self):
        """Test the fit method of PitmanYorProcess."""
        pyp = PitmanYorProcess(alpha=1.0, sigma=0.5)

        # Patch the assign_cluster method to avoid randomness
        with patch.object(pyp, "assign_cluster") as mock_assign:
            texts = ["text1", "text2", "text3"]
            clusters, params = pyp.fit(texts)

            assert mock_assign.call_count == 3

            assert clusters is pyp.clusters
            assert params is pyp.cluster_params
