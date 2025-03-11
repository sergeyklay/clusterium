"""Unit tests for the DirichletProcess and PitmanYorProcess classes."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from qadst.clustering.cache import EmbeddingCache
from qadst.clustering.models import DirichletProcess, PitmanYorProcess


@pytest.fixture
def mock_sentence_transformer():
    """Create a mock SentenceTransformer."""
    with patch("qadst.clustering.models.SentenceTransformer") as mock_st:
        # Configure the mock to return a fixed embedding when encode is called
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        mock_st.return_value = mock_instance
        yield mock_st


@pytest.fixture
def mock_embedding_cache():
    """Create a mock EmbeddingCache."""
    mock_cache = MagicMock(spec=EmbeddingCache)
    mock_cache.__contains__.return_value = False
    mock_cache.get.return_value = None
    return mock_cache


@pytest.fixture
def sample_embedding():
    """Return a sample embedding tensor."""
    return torch.tensor([0.1, 0.2, 0.3, 0.4])


class TestDirichletProcess:
    """Tests for the DirichletProcess class."""

    def test_init(self, mock_embedding_cache):
        """Test initialization of DirichletProcess."""
        dp = DirichletProcess(alpha=1.0, cache=mock_embedding_cache)

        assert dp.alpha == 1.0
        assert dp.clusters == []
        assert dp.cluster_params == []
        assert dp.cache == mock_embedding_cache
        assert dp.similarity_metric == dp.bert_similarity

    @patch("qadst.clustering.models.SentenceTransformer")
    def test_get_embedding_new(self, mock_st, mock_embedding_cache):
        """Test getting a new embedding."""
        # Configure the mock
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        mock_st.return_value = mock_instance

        dp = DirichletProcess(alpha=1.0, cache=mock_embedding_cache)
        embedding = dp.get_embedding("test text")

        # Verify the model was called to encode the text
        mock_instance.encode.assert_called_once_with("test text")
        # Verify the cache was checked
        mock_embedding_cache.__contains__.assert_called_once()
        # Verify the embedding was stored in cache
        mock_embedding_cache.set.assert_called_once()

        assert np.array_equal(embedding, np.array([0.1, 0.2, 0.3, 0.4]))

    def test_get_embedding_from_cache(self, mock_embedding_cache, sample_embedding):
        """Test getting an embedding from cache."""
        # Configure the mock to return an embedding
        mock_embedding_cache.__contains__.return_value = True
        mock_embedding_cache.get.return_value = sample_embedding

        dp = DirichletProcess(alpha=1.0, cache=mock_embedding_cache)
        embedding = dp.get_embedding("test text")

        # Verify the cache was checked
        mock_embedding_cache.__contains__.assert_called_once()
        mock_embedding_cache.get.assert_called_once()

        assert torch.equal(embedding, sample_embedding)

    def test_save_embedding_cache(self, mock_embedding_cache):
        """Test saving the embedding cache."""
        dp = DirichletProcess(alpha=1.0, cache=mock_embedding_cache)
        dp.save_embedding_cache()

        mock_embedding_cache.save_cache.assert_called_once()

    def test_bert_similarity(self, mock_sentence_transformer):
        """Test the bert_similarity method."""
        dp = DirichletProcess(alpha=1.0)

        # Create two embeddings with known cosine similarity
        text_embedding = np.array([1.0, 0.0, 0.0, 0.0])
        cluster_embedding = np.array([0.0, 1.0, 0.0, 0.0])

        # Patch the get_embedding method to return the text_embedding
        with patch.object(dp, "get_embedding", return_value=text_embedding):
            similarity = dp.bert_similarity("test text", cluster_embedding)

            # Cosine similarity between orthogonal vectors is 0, so 1 - 0 = 1
            assert similarity == 0.0

    @patch("numpy.random.choice")
    def test_assign_cluster_new(self, mock_choice, mock_sentence_transformer):
        """Test assigning a text to a new cluster."""
        # Configure the mock to choose a new cluster
        mock_choice.return_value = 0  # This will be the index of the new cluster

        dp = DirichletProcess(alpha=1.0)
        # Empty cluster params means any choice will create a new cluster

        sample_embedding = torch.tensor([0.1, 0.2, 0.3, 0.4])
        with patch.object(dp, "sample_new_cluster", return_value=sample_embedding):
            dp.assign_cluster("test text")

            assert len(dp.clusters) == 1
            assert len(dp.cluster_params) == 1
            assert dp.clusters[0] == 0
            assert torch.equal(dp.cluster_params[0], sample_embedding)

    @patch("numpy.random.choice")
    def test_assign_cluster_existing(self, mock_choice, mock_sentence_transformer):
        """Test assigning a text to an existing cluster."""
        dp = DirichletProcess(alpha=1.0)

        # Add a cluster parameter
        dp.cluster_params.append(torch.tensor([0.1, 0.2, 0.3, 0.4]))

        # Configure the mock to choose the existing cluster
        mock_choice.return_value = 0

        dp.assign_cluster("test text")

        assert len(dp.clusters) == 1
        assert len(dp.cluster_params) == 1
        assert dp.clusters[0] == 0

    def test_fit(self, mock_sentence_transformer):
        """Test the fit method."""
        dp = DirichletProcess(alpha=1.0)

        # Patch the assign_cluster method to avoid randomness
        with patch.object(dp, "assign_cluster") as mock_assign:
            with patch.object(dp, "save_embedding_cache") as mock_save:
                texts = ["text1", "text2", "text3"]
                clusters, params = dp.fit(texts)

                # Verify assign_cluster was called for each text
                assert mock_assign.call_count == 3
                # Verify save_embedding_cache was called
                mock_save.assert_called_once()

                # The clusters and params should be the ones from the
                # DirichletProcess instance
                assert clusters is dp.clusters
                assert params is dp.cluster_params


class TestPitmanYorProcess:
    """Tests for the PitmanYorProcess class."""

    def test_init(self, mock_embedding_cache):
        """Test initialization of PitmanYorProcess."""
        pyp = PitmanYorProcess(alpha=1.0, sigma=0.5, cache=mock_embedding_cache)

        assert pyp.alpha == 1.0
        assert pyp.sigma == 0.5
        assert pyp.clusters == []
        assert pyp.cluster_params == []
        assert pyp.cache == mock_embedding_cache
        assert pyp.similarity_metric == pyp.bert_similarity
        assert pyp.cluster_sizes == {}

    @patch("numpy.random.choice")
    @patch("qadst.clustering.models.cosine", return_value=0.2)  # 1 - 0.2 = 0.8
    def test_assign_cluster_new(
        self, mock_cosine, mock_choice, mock_sentence_transformer
    ):
        """Test assigning a text to a new cluster in PitmanYorProcess."""
        # Configure the mock to choose a new cluster
        mock_choice.return_value = 0  # This will be the index of the new cluster

        pyp = PitmanYorProcess(alpha=1.0, sigma=0.5)
        # Empty cluster params means any choice will create a new cluster

        sample_embedding = torch.tensor([0.1, 0.2, 0.3, 0.4])
        with patch.object(pyp, "get_embedding", return_value=sample_embedding):
            pyp.assign_cluster("test text")

            assert len(pyp.clusters) == 1
            assert len(pyp.cluster_params) == 1
            assert pyp.clusters[0] == 0
            assert torch.equal(pyp.cluster_params[0], sample_embedding)
            assert pyp.cluster_sizes == {0: 1}

    @patch("numpy.random.choice")
    @patch("qadst.clustering.models.cosine", return_value=0.2)  # 1 - 0.2 = 0.8
    def test_assign_cluster_existing(
        self, mock_cosine, mock_choice, mock_sentence_transformer
    ):
        """Test assigning a text to an existing cluster in PitmanYorProcess."""
        pyp = PitmanYorProcess(alpha=1.0, sigma=0.5)

        # Add a cluster parameter and update cluster_sizes
        pyp.cluster_params.append(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        pyp.clusters.append(0)
        pyp.cluster_sizes = {0: 1}

        # Configure the mock to choose the existing cluster
        mock_choice.return_value = 0

        sample_embedding = torch.tensor([0.5, 0.6, 0.7, 0.8])
        with patch.object(pyp, "get_embedding", return_value=sample_embedding):
            pyp.assign_cluster("test text")

            assert len(pyp.clusters) == 2
            assert len(pyp.cluster_params) == 1
            assert pyp.clusters == [0, 0]
            assert pyp.cluster_sizes == {0: 2}

    def test_fit(self, mock_sentence_transformer):
        """Test the fit method of PitmanYorProcess."""
        pyp = PitmanYorProcess(alpha=1.0, sigma=0.5)

        # Patch the assign_cluster method to avoid randomness
        with patch.object(pyp, "assign_cluster") as mock_assign:
            with patch.object(pyp, "save_embedding_cache") as mock_save:
                texts = ["text1", "text2", "text3"]
                clusters, params = pyp.fit(texts)

                # Verify assign_cluster was called for each text
                assert mock_assign.call_count == 3
                # Verify save_embedding_cache was called
                mock_save.assert_called_once()

                # The clusters and params should be the ones from the
                # PitmanYorProcess instance
                assert clusters is pyp.clusters
                assert params is pyp.cluster_params
