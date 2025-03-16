"""Unit tests for the DirichletProcess and PitmanYorProcess classes."""

# pylint: disable=protected-access

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from clusx.clustering.models import DirichletProcess, PitmanYorProcess


def test_dp_initialization(dp_instance):
    """Test initialization of DirichletProcess."""
    assert dp_instance.alpha == 1.0
    assert dp_instance.kappa == 1.0
    assert isinstance(dp_instance.clusters, list)
    assert isinstance(dp_instance.cluster_params, dict)
    assert dp_instance.embedding_dim is None
    assert dp_instance.next_id == 0


def test_dp_get_embedding(transformer_mock, embedding_fx, dp_instance):
    """Test getting embeddings from text."""
    mock_instance = transformer_mock.return_value
    mock_instance.encode.return_value = np.array([embedding_fx])

    with patch.object(dp_instance, "_normalize", side_effect=lambda x: x):
        embedding = dp_instance.get_embedding("test text")

        mock_instance.encode.assert_called_once_with(
            ["test text"], show_progress_bar=False
        )

        assert np.array_equal(embedding, embedding_fx)
        assert "test text" in dp_instance.text_embeddings


def test_dp_get_embedding_caching(transformer_mock, embedding_fx, dp_instance):
    """Test that embeddings are cached."""
    mock_instance = transformer_mock.return_value
    mock_instance.encode.return_value = np.array([embedding_fx])

    dp_instance.get_embedding("test text")
    dp_instance.get_embedding("test text")

    mock_instance.encode.assert_called_once_with(["test text"], show_progress_bar=False)


def test_dp_get_embedding_list(transformer_mock, embedding_fx, dp_instance):
    """Test getting embeddings for a list of texts."""
    texts = ["text1", "text2"]

    mock_instance = transformer_mock.return_value
    mock_instance.encode.return_value = np.array([embedding_fx, embedding_fx])

    embeddings = dp_instance.get_embedding(texts)
    mock_instance.encode.assert_called_once_with(texts, show_progress_bar=False)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 4)


def test_dp_normalize(dp_instance):
    """Test vector normalization."""
    vector = np.array([3.0, 4.0])
    normalized = dp_instance._normalize(vector)

    assert np.isclose(np.linalg.norm(normalized), 1.0)
    assert np.isclose(normalized[0], 0.6)
    assert np.isclose(normalized[1], 0.8)


def test_dp_log_likelihood_vmf(dp_instance, embedding_fx):
    """Test log likelihood calculation for a cluster."""
    cluster_id = 0
    dp_instance.cluster_params[cluster_id] = {"mean": embedding_fx, "count": 1}
    likelihood = dp_instance._log_likelihood_vmf(embedding_fx, cluster_id)

    assert isinstance(likelihood, float)
    assert likelihood > 0

    orthogonal = np.array([0.0, 0.0, 0.0, 1.0])
    likelihood_orthogonal = dp_instance._log_likelihood_vmf(orthogonal, cluster_id)

    assert likelihood_orthogonal < likelihood


def test_dp_log_likelihood_vmf_nonexistent_cluster(dp_instance, embedding_fx):
    """Test log likelihood for a nonexistent cluster."""
    dp_instance.global_mean = embedding_fx
    likelihood = dp_instance._log_likelihood_vmf(embedding_fx, 999)

    assert isinstance(likelihood, float)

    dp_instance.global_mean = None
    likelihood = dp_instance._log_likelihood_vmf(embedding_fx, 999)
    assert likelihood == 0.0


def test_dp_log_crp_prior(dp_instance):
    """Test Chinese Restaurant Process prior calculation."""
    prior_new = dp_instance.log_crp_prior()
    assert np.isclose(prior_new, np.log(1.0))

    dp_instance.clusters = [0, 0]
    dp_instance.cluster_params[0] = {"mean": np.array([0.1, 0.2]), "count": 2}

    prior_existing = dp_instance.log_crp_prior(0)
    prior_new = dp_instance.log_crp_prior()

    assert np.isclose(prior_existing, np.log(2 / 3))
    assert np.isclose(prior_new, np.log(1 / 3))


def test_dp_log_likelihood(dp_instance, embedding_fx):
    """Test log likelihood calculation across all clusters."""
    # Set up clusters
    dp_instance.cluster_params[0] = {"mean": embedding_fx, "count": 1}
    dp_instance.cluster_params[1] = {"mean": np.array([0.5, 0.5, 0.5, 0.5]), "count": 1}

    likelihoods, new_likelihood = dp_instance.log_likelihood(embedding_fx)

    assert 0 in likelihoods
    assert 1 in likelihoods

    assert isinstance(likelihoods[0], float)
    assert isinstance(likelihoods[1], float)
    assert new_likelihood == 0.0


def test_dp_calculate_cluster_probabilities(dp_instance, embedding_fx):
    """Test calculation of cluster assignment probabilities."""
    dp_instance.cluster_params[0] = {"mean": embedding_fx, "count": 1}
    dp_instance.clusters = [0]

    cluster_ids, probs = dp_instance._calculate_cluster_probabilities(embedding_fx)

    assert len(cluster_ids) == 2
    assert len(probs) == 2
    assert cluster_ids[0] == 0
    assert cluster_ids[1] is None
    assert np.isclose(np.sum(probs), 1.0)


def test_dp_create_new_cluster(dp_instance, embedding_fx):
    """Test creating a new cluster."""
    cluster_id = dp_instance._create_or_update_cluster(embedding_fx, True)

    assert cluster_id == 0
    assert len(dp_instance.clusters) == 1
    assert dp_instance.clusters[0] == 0
    assert dp_instance.cluster_params[0]["count"] == 1
    assert np.array_equal(dp_instance.cluster_params[0]["mean"], embedding_fx)
    assert dp_instance.next_id == 1


def test_dp_update_existing_cluster(dp_instance, embedding_fx):
    """Test updating an existing cluster."""
    dp_instance.cluster_params[0] = {"mean": embedding_fx, "count": 1}

    new_embedding = np.array([0.2, 0.3, 0.4, 0.5])
    cluster_id = dp_instance._create_or_update_cluster(new_embedding, False, 0)

    assert cluster_id == 0
    assert dp_instance.cluster_params[0]["count"] == 2

    expected_mean = dp_instance._normalize(embedding_fx + new_embedding)
    assert np.allclose(dp_instance.cluster_params[0]["mean"], expected_mean)


def test_dp_assign_cluster(dp_instance, embedding_fx):
    """Test assigning a document to a cluster."""
    with (
        patch.object(dp_instance, "_calculate_cluster_probabilities") as mock_probs,
        patch.object(dp_instance, "_create_or_update_cluster") as mock_create_update,
    ):
        mock_probs.return_value = ([0], np.array([1.0]))

        dp_instance.random_state = MagicMock()
        dp_instance.random_state.choice.return_value = 0

        mock_create_update.return_value = 0
        dp_instance.cluster_params[0] = {"mean": embedding_fx, "count": 1}

        cluster_id, probs = dp_instance.assign_cluster(embedding_fx)

        assert cluster_id == 0
        assert np.array_equal(probs, np.array([1.0]))

        mock_create_update.assert_called_once()


def test_dp_assign_new_cluster(dp_instance, embedding_fx):
    """Test assigning a document to a new cluster."""
    with patch.object(dp_instance, "_calculate_cluster_probabilities") as mock_probs:
        mock_probs.return_value = ([None], np.array([1.0]))

        dp_instance.random_state = MagicMock()
        dp_instance.random_state.choice.return_value = 0

        cluster_id, probs = dp_instance.assign_cluster(embedding_fx)

        assert cluster_id == 0  # First new cluster gets ID 0
        assert np.array_equal(probs, np.array([1.0]))
        assert dp_instance.cluster_params[0]["count"] == 1
        assert np.array_equal(dp_instance.cluster_params[0]["mean"], embedding_fx)


def test_dp_fit(dp_instance, sample_texts):
    """Test fitting the model to text data."""
    with (
        patch.object(dp_instance, "get_embedding") as mock_embed,
        patch.object(dp_instance, "assign_cluster") as mock_assign,
    ):
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        mock_embed.return_value = mock_embeddings

        mock_assign.side_effect = [(i, None) for i in range(len(sample_texts))]

        dp_instance.fit(sample_texts)

        assert mock_embed.call_count == 1
        assert mock_assign.call_count == len(sample_texts)
        assert np.array_equal(dp_instance.embeddings_, mock_embeddings)
        assert np.array_equal(dp_instance.labels_, np.array([0, 1, 2]))


def test_dp_predict(dp_instance, embedding_fx):
    """Test predicting clusters for new data."""
    dp_instance.cluster_params[0] = {"mean": embedding_fx, "count": 1}
    dp_instance.cluster_params[1] = {"mean": np.array([0.9, 0.8, 0.7, 0.6]), "count": 1}

    with (
        patch.object(dp_instance, "get_embedding") as mock_embed,
        patch.object(dp_instance, "_log_likelihood_vmf") as mock_likelihood,
    ):
        mock_embed.return_value = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.9, 0.8, 0.7, 0.6],
            ]
        )

        def likelihood_side_effect(emb, cid):
            if np.array_equal(emb, np.array([0.1, 0.2, 0.3, 0.4])):
                return 0.9 if cid == 0 else 0.1
            else:
                return 0.1 if cid == 0 else 0.9

        mock_likelihood.side_effect = likelihood_side_effect
        predictions = dp_instance.predict(["text1", "text2"])

        assert np.array_equal(predictions, np.array([0, 1]))


def test_dp_fit_predict(dp_instance, sample_texts):
    """Test fit_predict method."""
    with patch.object(dp_instance, "fit") as mock_fit:
        dp_instance.labels_ = np.array([0, 1, 2])
        dp_instance.fit_predict(sample_texts)

        mock_fit.assert_called_once_with(sample_texts)
        assert np.array_equal(dp_instance.labels_, np.array([0, 1, 2]))


def test_pyp_initialization(pyp_instance):
    """Test initialization of PitmanYorProcess."""
    assert pyp_instance.alpha == 1.0
    assert pyp_instance.kappa == 1.0
    assert pyp_instance.sigma == 0.5
    assert isinstance(pyp_instance.clusters, list)
    assert isinstance(pyp_instance.cluster_params, dict)


def test_pyp_initialization_invalid_sigma():
    """Test initialization with invalid sigma values."""
    with pytest.raises(ValueError) as excinfo:
        PitmanYorProcess(alpha=1.0, kappa=1.0, sigma=1.0)
    assert "sigma must be in the interval [0.0, 1.0)" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        PitmanYorProcess(alpha=1.0, kappa=1.0, sigma=-0.1)
    assert "sigma must be in the interval [0.0, 1.0)" in str(excinfo.value)


def test_pyp_initialization_invalid_alpha():
    """Test initialization with invalid alpha values."""
    with pytest.raises(ValueError) as excinfo:
        PitmanYorProcess(alpha=-0.6, kappa=1.0, sigma=0.5)
    assert "alpha must be greater than -sigma" in str(excinfo.value)


def test_pyp_log_pyp_prior_empty(pyp_instance):
    """Test PYP prior calculation with no documents."""
    prior = pyp_instance.log_pyp_prior()
    assert prior == 0.0  # Log of 1.0


def test_pyp_log_pyp_prior(pyp_instance):
    """Test Pitman-Yor Process prior calculation."""
    pyp_instance.clusters = [0, 0]
    pyp_instance.cluster_params[0] = {"mean": np.array([0.1, 0.2]), "count": 2}

    prior_existing = pyp_instance.log_pyp_prior(0)
    prior_new = pyp_instance.log_pyp_prior()

    assert np.isclose(prior_existing, np.log(1.5 / 3))
    assert np.isclose(prior_new, np.log(1.5 / 3))


def test_pyp_log_pyp_prior_small_count(pyp_instance):
    """Test PYP prior with count smaller than sigma."""
    pyp_instance.clusters = [0]
    pyp_instance.cluster_params[0] = {"mean": np.array([0.1, 0.2]), "count": 0.3}

    prior = pyp_instance.log_pyp_prior(0)
    assert prior < 0


def test_pyp_calculate_cluster_probabilities(pyp_instance, embedding_fx):
    """Test calculation of cluster probabilities using PYP."""
    pyp_instance.cluster_params[0] = {"mean": embedding_fx, "count": 1}
    pyp_instance.clusters = [0]

    with (
        patch.object(pyp_instance, "log_pyp_prior") as mock_prior,
        patch.object(pyp_instance, "log_likelihood") as mock_likelihood,
    ):
        mock_prior.side_effect = [np.log(0.4), np.log(0.6)]
        mock_likelihood.return_value = ({0: 1.0}, 0.5)

        cluster_ids, probs = pyp_instance._calculate_cluster_probabilities(embedding_fx)

        assert len(cluster_ids) == 2
        assert cluster_ids[0] == 0
        assert cluster_ids[1] is None

        expected_unnormalized = [np.exp(np.log(0.4) + 1.0), np.exp(np.log(0.6) + 0.5)]
        expected_sum = sum(expected_unnormalized)
        expected_probs = [p / expected_sum for p in expected_unnormalized]

        assert np.allclose(probs, expected_probs)
        assert np.isclose(np.sum(probs), 1.0)


def test_pyp_assign_cluster(pyp_instance, embedding_fx):
    """Test assigning a document to a cluster in PYP."""
    with (
        patch.object(pyp_instance, "_calculate_cluster_probabilities") as mock_probs,
        patch.object(pyp_instance, "_create_or_update_cluster") as mock_create_update,
    ):
        mock_probs.return_value = ([0], np.array([1.0]))

        pyp_instance.random_state = MagicMock()
        pyp_instance.random_state.choice.return_value = 0

        mock_create_update.return_value = 0
        pyp_instance.cluster_params[0] = {"mean": embedding_fx, "count": 1}

        cluster_id, probs = pyp_instance.assign_cluster(embedding_fx)

        assert cluster_id == 0
        assert np.array_equal(probs, np.array([1.0]))

        mock_create_update.assert_called_once()


def test_pyp_assign_new_cluster(pyp_instance, embedding_fx):
    """Test assigning a document to a new cluster in PYP."""
    with patch.object(pyp_instance, "_calculate_cluster_probabilities") as mock_probs:
        mock_probs.return_value = ([None], np.array([1.0]))
        pyp_instance.random_state = MagicMock()
        pyp_instance.random_state.choice.return_value = 0

        cluster_id, probs = pyp_instance.assign_cluster(embedding_fx)

        assert cluster_id == 0
        assert np.array_equal(probs, np.array([1.0]))
        assert pyp_instance.cluster_params[0]["count"] == 1
        assert np.array_equal(pyp_instance.cluster_params[0]["mean"], embedding_fx)


def test_pyp_fit(pyp_instance, sample_texts):
    """Test fitting the PYP model to text data."""
    with (
        patch.object(pyp_instance, "get_embedding") as mock_embed,
        patch.object(pyp_instance, "assign_cluster") as mock_assign,
    ):
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        mock_embed.return_value = mock_embeddings

        mock_assign.side_effect = [(i, None) for i in range(len(sample_texts))]

        pyp_instance.fit(sample_texts)

        assert mock_embed.call_count == 1
        assert mock_assign.call_count == len(sample_texts)
        assert np.array_equal(pyp_instance.embeddings_, mock_embeddings)
        assert np.array_equal(pyp_instance.labels_, np.array([0, 1, 2]))


def test_pyp_predict(pyp_instance, embedding_fx):
    """Test predicting clusters for new data with PYP."""
    pyp_instance.cluster_params[0] = {"mean": embedding_fx, "count": 1}
    pyp_instance.cluster_params[1] = {
        "mean": np.array([0.9, 0.8, 0.7, 0.6]),
        "count": 1,
    }

    with (
        patch.object(pyp_instance, "get_embedding") as mock_embed,
        patch.object(pyp_instance, "_log_likelihood_vmf") as mock_likelihood,
    ):
        mock_embed.return_value = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.9, 0.8, 0.7, 0.6],
            ]
        )

        def likelihood_side_effect(emb, cid):
            if np.array_equal(emb, np.array([0.1, 0.2, 0.3, 0.4])):
                return 0.9 if cid == 0 else 0.1
            else:
                return 0.1 if cid == 0 else 0.9

        mock_likelihood.side_effect = likelihood_side_effect

        predictions = pyp_instance.predict(["text1", "text2"])

        assert np.array_equal(predictions, np.array([0, 1]))


def test_pyp_fit_predict(pyp_instance, sample_texts):
    """Test fit_predict method for PYP."""
    with patch.object(pyp_instance, "fit") as mock_fit:
        pyp_instance.labels_ = np.array([0, 1, 2])
        pyp_instance.fit_predict(sample_texts)

        mock_fit.assert_called_once_with(sample_texts)
        assert np.array_equal(pyp_instance.labels_, np.array([0, 1, 2]))


def test_pyp_inheritance(pyp_instance):
    """Test that PYP inherits and extends DP functionality."""
    assert hasattr(pyp_instance, "get_embedding")
    assert hasattr(pyp_instance, "_normalize")
    assert hasattr(pyp_instance, "log_likelihood")
    assert hasattr(pyp_instance, "fit")
    assert hasattr(pyp_instance, "predict")
    assert hasattr(pyp_instance, "log_pyp_prior")
    assert (
        pyp_instance._calculate_cluster_probabilities.__func__
        != DirichletProcess._calculate_cluster_probabilities
    )


def test_pyp_vs_dp_behavior(dp_instance, pyp_instance, embedding_fx):
    """Test that PYP and DP behave differently with the same input."""
    dp_instance.cluster_params[0] = {"mean": embedding_fx, "count": 1}
    dp_instance.clusters = [0]

    pyp_instance.cluster_params[0] = {"mean": embedding_fx, "count": 1}
    pyp_instance.clusters = [0]

    random_state = np.random.default_rng(seed=42)
    dp_instance.random_state = random_state
    pyp_instance.random_state = random_state

    assert hasattr(pyp_instance, "log_pyp_prior")
    assert not hasattr(dp_instance, "log_pyp_prior")
    assert hasattr(pyp_instance, "sigma")
    assert not hasattr(dp_instance, "sigma")
