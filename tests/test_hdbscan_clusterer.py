"""Unit tests for the HDBSCANQAClusterer class."""

import tempfile
from unittest.mock import patch

from qadst.clusterer import HDBSCANQAClusterer


def test_cluster_method(mock_hdbscan_clusterer):
    """Test the cluster_method method."""
    assert mock_hdbscan_clusterer.cluster_method() == "hdbscan"


def test_calculate_min_cluster_size():
    """Test the _calculate_min_cluster_size method."""
    with (
        patch("qadst.clusterer.HDBSCAN"),
        patch("qadst.base.OpenAIEmbeddings"),
        patch("qadst.base.ChatOpenAI"),
    ):
        clusterer = HDBSCANQAClusterer(
            embedding_model_name="test-model",
            output_dir=tempfile.mkdtemp(),
        )

        # Test with small dataset
        assert clusterer._calculate_min_cluster_size(30) == 11

        # Test with medium dataset
        assert clusterer._calculate_min_cluster_size(150) == 25

        # Test with large dataset
        assert clusterer._calculate_min_cluster_size(500) == 38

        # Test with very large dataset
        assert clusterer._calculate_min_cluster_size(3000) == 64

        # Test with extremely large dataset (should be capped at 100)
        assert clusterer._calculate_min_cluster_size(100000) == 100


def test_cluster_questions_empty_input(mock_hdbscan_clusterer):
    """Test the cluster_questions method with empty input."""
    result = mock_hdbscan_clusterer.cluster_questions([])
    assert result == {"clusters": []}


@patch("qadst.clusterer.HDBSCANQAClusterer._perform_hdbscan_clustering")
def test_cluster_questions_delegates_to_perform_hdbscan(
    mock_perform, mock_hdbscan_clusterer
):
    """Test that cluster_questions delegates to _perform_hdbscan_clustering."""
    # Setup mock return value
    expected_result = {"clusters": [{"id": 1, "representative": [], "source": []}]}
    mock_perform.return_value = expected_result

    # Call the method
    qa_pairs = [("test question", "test answer")]
    result = mock_hdbscan_clusterer.cluster_questions(qa_pairs)

    # Verify the mock was called with the right arguments
    mock_perform.assert_called_once_with(qa_pairs)

    # Verify the result
    assert result == expected_result
