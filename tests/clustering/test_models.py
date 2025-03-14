"""Unit tests for the DirichletProcess and PitmanYorProcess classes."""

from typing import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from clusx.clustering.models import DirichletProcess, PitmanYorProcess


def test_dp_simple_init(dp_instance):
    """Test simple initialization of DirichletProcess."""
    dp = dp_instance

    assert dp.alpha == 1.0
    assert not hasattr(dp, "sigma")

    assert isinstance(dp.clusters, list)
    assert isinstance(dp.cluster_params, dict)
    assert isinstance(dp.base_measure, dict)

    assert isinstance(dp.similarity_metric, Callable)
    assert dp.similarity_metric.__name__ == "cosine_similarity"


def test_dp_base_measure_default(dp_instance):
    """Test that DirichletProcess uses default base_measure when none is provided."""
    dp = dp_instance

    assert dp.base_measure == {"variance": 0.3}


def test_dp_base_measure_custom():
    """Test that DirichletProcess accepts custom base_measure."""
    custom_base_measure = {"variance": 0.5}
    dp = DirichletProcess(alpha=1.0, base_measure=custom_base_measure)

    assert dp.base_measure == custom_base_measure
    assert dp.base_measure["variance"] == 0.5


def test_dp_base_measure_invalid_type():
    """Test that DirichletProcess raises TypeError when variance is not a float."""
    with pytest.raises(TypeError) as excinfo:
        DirichletProcess(alpha=1.0, base_measure={"variance": "not_a_float"})

    assert "variance in base_measure must be a float" in str(excinfo.value)


def test_dp_get_embedding(
    transformer_mock: MagicMock, embedding_fx: np.ndarray, dp_instance
):
    """Test getting a new embedding."""
    dp = dp_instance
    embedding = dp.get_embedding("test text")

    mock_instance = transformer_mock.return_value
    mock_instance.encode.assert_called_once_with("test text", show_progress_bar=False)

    assert np.array_equal(embedding, embedding_fx)


def test_dp_cosine_similarity(dp_instance):
    """Test the cosine_similarity method."""
    dp = dp_instance

    text_embedding = np.asarray([1.0, 0.0, 0.0, 0.0])
    cluster_embedding = np.asarray([0.0, 1.0, 0.0, 0.0])
    similarity = dp.cosine_similarity(text_embedding, cluster_embedding)

    assert similarity == 0.0


def test_dp_log_likelihood_base_measure(dp_instance, embedding_fx: np.ndarray):
    """Test the _log_likelihood_base_measure method of DirichletProcess."""
    dp = dp_instance

    dp.embedding_dim = len(embedding_fx)

    variance = dp.base_measure["variance"] * 10.0
    dim = float(dp.embedding_dim)
    expected_log_likelihood = -0.5 * dim * np.log(2 * np.pi * variance)

    actual_log_likelihood = dp._log_likelihood_base_measure(embedding_fx)

    assert np.isclose(actual_log_likelihood, expected_log_likelihood)

    dp.base_measure = {"variance": 0.5}
    variance = dp.base_measure["variance"] * 10.0
    expected_log_likelihood = -0.5 * dim * np.log(2 * np.pi * variance)
    actual_log_likelihood = dp._log_likelihood_base_measure(embedding_fx)
    assert np.isclose(actual_log_likelihood, expected_log_likelihood)


def test_dp_assign_cluster_new(embedding_fx: np.ndarray, dp_instance):
    """Test assigning a text to a new cluster."""
    dp = dp_instance

    with patch.object(dp, "get_embedding", return_value=embedding_fx):
        cluster_id = dp.assign_cluster("test text")

        assert len(dp.clusters) == 1
        assert len(dp.cluster_params) == 1
        assert dp.clusters[0] == 0
        assert cluster_id == 0
        assert 0 in dp.cluster_params


def test_dp_assign_cluster_existing(embedding_fx: np.ndarray, dp_instance):
    """Test assigning a text to an existing cluster."""
    dp = dp_instance

    dp.cluster_params[0] = {"mean": embedding_fx, "count": 1}
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


def test_dp_fit(dp_instance):
    """Test the fit method of DirichletProcess."""
    dp = dp_instance
    with patch.object(dp, "assign_cluster") as mock_assign:
        texts = ["text1", "text2", "text3"]
        clusters, params = dp.fit(texts)

        assert mock_assign.call_count == 3
        assert clusters is dp.clusters
        assert params is dp.cluster_params


def test_dp_fit_with_sample_texts(dp_instance, sample_texts):
    """Test fitting the DirichletProcess model with sample texts from fixtures."""
    dp = dp_instance

    dp.clusters = []
    dp.cluster_params = {}

    original_assign_cluster = dp.assign_cluster
    try:

        def mock_side_effect(_):
            dp.clusters.append(0)
            return 0

        mock_assign = MagicMock(side_effect=mock_side_effect)
        dp.assign_cluster = mock_assign

        clusters, _ = dp.fit(sample_texts)

        assert mock_assign.call_count == len(sample_texts)
        assert len(clusters) == len(sample_texts)
    finally:
        dp.assign_cluster = original_assign_cluster


def test_pyp_init(pyp_instance):
    """Test initialization of PitmanYorProcess."""
    pyp = pyp_instance

    assert pyp.alpha == 1.0
    assert pyp.sigma == 0.5
    assert isinstance(pyp.clusters, list)
    assert len(pyp.clusters) == 0
    assert isinstance(pyp.cluster_params, dict)
    assert pyp.similarity_metric == pyp.cosine_similarity


def test_pyp_base_measure_default(pyp_instance):
    """Test that PitmanYorProcess uses default base_measure when none is provided."""
    pyp = pyp_instance

    assert pyp.base_measure == {"variance": 0.3}


def test_pyp_base_measure_custom():
    """Test that PitmanYorProcess accepts custom base_measure."""
    custom_base_measure = {"variance": 0.5}
    pyp = PitmanYorProcess(alpha=1.0, sigma=0.5, base_measure=custom_base_measure)

    assert pyp.base_measure == custom_base_measure
    assert pyp.base_measure["variance"] == 0.5


def test_pyp_base_measure_invalid_type():
    """Test that PitmanYorProcess raises TypeError when variance is not a float."""
    with pytest.raises(TypeError) as excinfo:
        PitmanYorProcess(alpha=1.0, sigma=0.5, base_measure={"variance": "not_a_float"})

    assert "variance in base_measure must be a float" in str(excinfo.value)


def test_pyp_log_likelihood_base_measure(pyp_instance, embedding_fx: np.ndarray):
    """Test the _log_likelihood_base_measure method of PitmanYorProcess."""
    pyp = pyp_instance

    pyp.embedding_dim = len(embedding_fx)

    variance = pyp.base_measure["variance"] * 10.0
    dim = float(pyp.embedding_dim)
    expected_log_likelihood = -0.5 * dim * np.log(2 * np.pi * variance)

    actual_log_likelihood = pyp._log_likelihood_base_measure(embedding_fx)

    assert np.isclose(actual_log_likelihood, expected_log_likelihood)

    pyp.base_measure = {"variance": 0.5}
    variance = pyp.base_measure["variance"] * 10.0
    expected_log_likelihood = -0.5 * dim * np.log(2 * np.pi * variance)
    actual_log_likelihood = pyp._log_likelihood_base_measure(embedding_fx)
    assert np.isclose(actual_log_likelihood, expected_log_likelihood)


def test_pyp_assign_cluster_new(embedding_fx: np.ndarray, pyp_instance):
    """Test assigning a text to a new cluster in PitmanYorProcess."""
    pyp = pyp_instance

    with patch.object(pyp, "get_embedding", return_value=embedding_fx):
        cluster_id = pyp.assign_cluster("test text")

        assert len(pyp.clusters) == 1
        assert len(pyp.cluster_params) == 1
        assert pyp.clusters[0] == 0
        assert cluster_id == 0
        assert 0 in pyp.cluster_params


def test_pyp_assign_cluster_existing(embedding_fx: np.ndarray, pyp_instance):
    """Test assigning a text to an existing cluster in PitmanYorProcess."""
    pyp = pyp_instance

    pyp.cluster_params[0] = {"mean": embedding_fx, "count": 1}
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


def test_pyp_fit(pyp_instance):
    """Test the fit method of PitmanYorProcess."""
    pyp = pyp_instance
    with patch.object(pyp, "assign_cluster") as mock_assign:
        texts = ["text1", "text2", "text3"]
        clusters, params = pyp.fit(texts)

        assert mock_assign.call_count == 3
        assert clusters is pyp.clusters
        assert params is pyp.cluster_params


def test_pyp_fit_with_sample_texts(pyp_instance, sample_texts):
    """Test fitting the PitmanYorProcess model with sample texts from fixtures."""
    pyp = pyp_instance

    pyp.clusters = []
    pyp.cluster_params = {}

    original_assign_cluster = pyp.assign_cluster
    try:

        def mock_side_effect(_):
            pyp.clusters.append(0)
            return 0

        mock_assign = MagicMock(side_effect=mock_side_effect)
        pyp.assign_cluster = mock_assign
        clusters, _ = pyp.fit(sample_texts)

        assert mock_assign.call_count == len(sample_texts)
        assert len(clusters) == len(sample_texts)
    finally:
        pyp.assign_cluster = original_assign_cluster


def test_pyp_cluster_sizes_tracking(
    embedding_fx: np.ndarray, pyp_instance, sample_texts
):
    """Test that cluster parameters are properly tracked in PitmanYorProcess."""
    pyp = pyp_instance

    pyp.clusters = []
    pyp.cluster_params = {}

    with patch.object(pyp, "get_embedding", return_value=embedding_fx):
        cluster_id = pyp.assign_cluster(sample_texts[0])
        assert cluster_id == 0

    original_assign_cluster = pyp.assign_cluster
    test_embedding = np.array([0.5, 0.6, 0.7, 0.8])

    try:
        with patch.object(pyp, "get_embedding", return_value=test_embedding):

            def create_cluster_1(_):
                pyp.clusters.append(1)
                pyp.cluster_params[1] = {"mean": test_embedding, "count": 1}
                return 1

            pyp.assign_cluster = MagicMock(side_effect=create_cluster_1)
            cluster_id = pyp.assign_cluster(sample_texts[1])
            assert cluster_id == 1

        with patch.object(pyp, "get_embedding", return_value=test_embedding):

            def add_to_cluster_1(_):
                pyp.clusters.append(1)
                pyp.cluster_params[1]["count"] += 1
                return 1

            pyp.assign_cluster = MagicMock(side_effect=add_to_cluster_1)
            cluster_id = pyp.assign_cluster(sample_texts[2])
            assert cluster_id == 1
    finally:
        pyp.assign_cluster = original_assign_cluster

    assert 0 in pyp.cluster_params
    assert 1 in pyp.cluster_params
    assert pyp.cluster_params[0]["count"] == 1
    assert pyp.cluster_params[1]["count"] == 2


def test_dp_base_measure_missing_variance():
    """Test default variance when base_measure lacks the variance key."""
    dp = DirichletProcess(alpha=1.0, base_measure={"other_key": "value"})

    assert dp.base_measure == {"variance": 0.3}
    assert "other_key" not in dp.base_measure


def test_pyp_base_measure_missing_variance():
    """Test default variance when base_measure lacks the variance key."""
    pyp = PitmanYorProcess(alpha=1.0, sigma=0.5, base_measure={"other_key": "value"})

    assert pyp.base_measure == {"variance": 0.3}
    assert "other_key" not in pyp.base_measure
