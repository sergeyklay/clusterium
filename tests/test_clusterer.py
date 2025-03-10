"""Tests for the HDBSCANQAClusterer class."""

import tempfile
from unittest.mock import MagicMock, patch

import numpy as np

from qadst.clusterer import HDBSCANQAClusterer


def test_cluster_method(mock_hdbscan_clusterer):
    """Test the cluster_method method."""
    assert mock_hdbscan_clusterer.cluster_method() == "hdbscan"


def test_calculate_min_cluster_size():
    """Test the _calculate_min_cluster_size method."""
    with (
        patch("qadst.clusterer.HDBSCAN"),
        patch("qadst.embeddings.get_embeddings_model"),
        patch("qadst.base.ChatOpenAI"),
    ):
        mock_embeddings_provider = MagicMock()
        mock_embeddings_provider.get_model_name.return_value = "test-model"

        clusterer = HDBSCANQAClusterer(
            embeddings_provider=mock_embeddings_provider,
            output_dir=tempfile.mkdtemp(),
        )

        clusterer.min_cluster_size = 15
        assert clusterer._calculate_min_cluster_size(100) == 15

        clusterer.min_cluster_size = None

        assert clusterer._calculate_min_cluster_size(10) == 5
        assert clusterer._calculate_min_cluster_size(100) == 21
        assert clusterer._calculate_min_cluster_size(1000) == 47
        assert clusterer._calculate_min_cluster_size(10000) == 84

        assert clusterer._calculate_min_cluster_size(100000) == 100


def test_identify_large_clusters(mock_hdbscan_clusterer):
    """Test the _identify_large_clusters method."""
    clusters = {
        "1": {
            "questions": ["q1", "q2", "q3", "q4", "q5"],  # 5 questions (small)
            "qa_pairs": [{"question": "q1", "answer": "a1"} for _ in range(5)],
        },
        "2": {
            "questions": ["q" + str(i) for i in range(1, 21)],  # 20 questions (medium)
            "qa_pairs": [
                {"question": f"q{i}", "answer": f"a{i}"} for i in range(1, 21)
            ],
        },
        "3": {
            "questions": ["q" + str(i) for i in range(1, 51)],  # 50 questions (large)
            "qa_pairs": [
                {"question": f"q{i}", "answer": f"a{i}"} for i in range(1, 51)
            ],
        },
        "4": {
            "questions": [
                "q" + str(i) for i in range(1, 101)
            ],  # 100 questions (very large)
            "qa_pairs": [
                {"question": f"q{i}", "answer": f"a{i}"} for i in range(1, 101)
            ],
        },
    }

    large_clusters = mock_hdbscan_clusterer._identify_large_clusters(clusters, 30)
    assert len(large_clusters) == 2
    assert "1" not in large_clusters
    assert "2" not in large_clusters
    assert "3" in large_clusters
    assert "4" in large_clusters

    large_clusters = mock_hdbscan_clusterer._identify_large_clusters(clusters, 50)
    assert len(large_clusters) == 1
    assert "1" not in large_clusters
    assert "2" not in large_clusters
    assert "3" not in large_clusters
    assert "4" in large_clusters

    large_clusters = mock_hdbscan_clusterer._identify_large_clusters(clusters, 100)
    assert len(large_clusters) == 0

    large_clusters = mock_hdbscan_clusterer._identify_large_clusters({}, 30)
    assert len(large_clusters) == 0


def test_cluster_questions_empty_input(mock_hdbscan_clusterer):
    """Test the cluster_questions method with empty input."""
    result = mock_hdbscan_clusterer.cluster_questions([])
    assert result == {"clusters": []}


@patch("qadst.clusterer.HDBSCANQAClusterer._perform_hdbscan_clustering")
def test_cluster_questions_delegates_to_perform_hdbscan(
    mock_perform, mock_hdbscan_clusterer
):
    """Test that cluster_questions delegates to _perform_hdbscan_clustering."""
    expected_result = {"clusters": [{"id": 1, "representative": [], "source": []}]}
    mock_perform.return_value = expected_result

    qa_pairs = [("test question", "test answer")]
    result = mock_hdbscan_clusterer.cluster_questions(qa_pairs)

    mock_perform.assert_called_once_with(qa_pairs)

    assert result == expected_result


def test_get_recursive_hdbscan_params(mock_hdbscan_clusterer):
    """Test the _get_recursive_hdbscan_params method."""
    min_cluster_size, epsilon = mock_hdbscan_clusterer._get_recursive_hdbscan_params(20)
    assert min_cluster_size == max(int(np.log(20) ** 1.5), 5)
    assert epsilon == 0.2

    min_cluster_size, epsilon = mock_hdbscan_clusterer._get_recursive_hdbscan_params(
        100
    )
    assert min_cluster_size == max(int(np.log(100) ** 1.5), 5)
    assert epsilon == 0.2

    min_cluster_size, epsilon = mock_hdbscan_clusterer._get_recursive_hdbscan_params(
        500
    )
    assert min_cluster_size == max(int(np.log(500) ** 1.5), 5)
    assert epsilon == 0.2

    min_cluster_size, epsilon = mock_hdbscan_clusterer._get_recursive_hdbscan_params(5)
    assert min_cluster_size == 5
    assert epsilon == 0.2

    expected_20 = max(int(np.log(20) ** 1.5), 5)
    expected_100 = max(int(np.log(100) ** 1.5), 5)
    expected_500 = max(int(np.log(500) ** 1.5), 5)
    expected_5 = 5

    assert mock_hdbscan_clusterer._get_recursive_hdbscan_params(20) == (
        expected_20,
        0.2,
    )
    assert mock_hdbscan_clusterer._get_recursive_hdbscan_params(100) == (
        expected_100,
        0.2,
    )
    assert mock_hdbscan_clusterer._get_recursive_hdbscan_params(500) == (
        expected_500,
        0.2,
    )
    assert mock_hdbscan_clusterer._get_recursive_hdbscan_params(5) == (expected_5, 0.2)


def test_calculate_kmeans_clusters(mock_hdbscan_clusterer):
    """Test the _calculate_kmeans_clusters method."""
    assert mock_hdbscan_clusterer._calculate_kmeans_clusters(10) == 2
    assert mock_hdbscan_clusterer._calculate_kmeans_clusters(30) == 2
    assert mock_hdbscan_clusterer._calculate_kmeans_clusters(59) == 2

    assert mock_hdbscan_clusterer._calculate_kmeans_clusters(60) == 2
    assert mock_hdbscan_clusterer._calculate_kmeans_clusters(90) == 3
    assert mock_hdbscan_clusterer._calculate_kmeans_clusters(150) == 5
    assert mock_hdbscan_clusterer._calculate_kmeans_clusters(240) == 8

    assert mock_hdbscan_clusterer._calculate_kmeans_clusters(300) == 10
    assert mock_hdbscan_clusterer._calculate_kmeans_clusters(500) == 10
    assert mock_hdbscan_clusterer._calculate_kmeans_clusters(1000) == 10

    for num_questions in [10, 60, 90, 150, 300, 500]:
        expected = min(max(2, int(num_questions / 30)), 10)
        assert (
            mock_hdbscan_clusterer._calculate_kmeans_clusters(num_questions) == expected
        )


def test_create_subclusters(mock_hdbscan_clusterer):
    """Test the _create_subclusters method."""
    cluster_id = "1"
    questions = ["q1", "q2", "q3", "q4", "q5"]
    qa_pairs = [
        {"question": "q1", "answer": "a1"},
        {"question": "q2", "answer": "a2"},
        {"question": "q3", "answer": "a3"},
        {"question": "q4", "answer": "a4"},
        {"question": "q5", "answer": "a5"},
    ]

    subcluster_labels = np.array([0, 0, 1, 1, 1])
    subclusters = mock_hdbscan_clusterer._create_subclusters(
        cluster_id, questions, qa_pairs, subcluster_labels
    )

    assert len(subclusters) == 2
    assert "1.0" in subclusters
    assert "1.1" in subclusters

    assert len(subclusters["1.0"]["questions"]) == 2
    assert subclusters["1.0"]["questions"] == ["q1", "q2"]
    assert len(subclusters["1.0"]["qa_pairs"]) == 2
    assert subclusters["1.0"]["qa_pairs"] == [
        {"question": "q1", "answer": "a1"},
        {"question": "q2", "answer": "a2"},
    ]

    assert len(subclusters["1.1"]["questions"]) == 3
    assert subclusters["1.1"]["questions"] == ["q3", "q4", "q5"]
    assert len(subclusters["1.1"]["qa_pairs"]) == 3
    assert subclusters["1.1"]["qa_pairs"] == [
        {"question": "q3", "answer": "a3"},
        {"question": "q4", "answer": "a4"},
        {"question": "q5", "answer": "a5"},
    ]

    subcluster_labels = np.array([0, -1, 1, -1, 1])
    subclusters = mock_hdbscan_clusterer._create_subclusters(
        cluster_id, questions, qa_pairs, subcluster_labels
    )

    assert len(subclusters) == 2
    assert "1.0" in subclusters
    assert "1.1" in subclusters

    assert len(subclusters["1.0"]["questions"]) == 1
    assert subclusters["1.0"]["questions"] == ["q1"]
    assert len(subclusters["1.0"]["qa_pairs"]) == 1
    assert subclusters["1.0"]["qa_pairs"] == [{"question": "q1", "answer": "a1"}]

    assert len(subclusters["1.1"]["questions"]) == 2
    assert subclusters["1.1"]["questions"] == ["q3", "q5"]
    assert len(subclusters["1.1"]["qa_pairs"]) == 2
    assert subclusters["1.1"]["qa_pairs"] == [
        {"question": "q3", "answer": "a3"},
        {"question": "q5", "answer": "a5"},
    ]

    subcluster_labels = np.array([-1, -1, -1, -1, -1])
    subclusters = mock_hdbscan_clusterer._create_subclusters(
        cluster_id, questions, qa_pairs, subcluster_labels
    )

    assert len(subclusters) == 0


@patch("qadst.clusterer.HDBSCANQAClusterer._identify_large_clusters")
@patch("qadst.clusterer.HDBSCANQAClusterer._apply_recursive_clustering")
@patch("qadst.clusterer.HDBSCANQAClusterer._create_subclusters")
def test_handle_large_clusters(
    mock_create_subclusters,
    mock_apply_clustering,
    mock_identify_large,
    mock_hdbscan_clusterer,
):
    """Test the _handle_large_clusters method."""
    clusters = {
        "1": {
            "questions": ["q1", "q2", "q3"],
            "qa_pairs": [
                {"question": "q1", "answer": "a1"},
                {"question": "q2", "answer": "a2"},
                {"question": "q3", "answer": "a3"},
            ],
        },
        "2": {
            "questions": ["q4", "q5", "q6", "q7", "q8"],
            "qa_pairs": [
                {"question": "q4", "answer": "a4"},
                {"question": "q5", "answer": "a5"},
                {"question": "q6", "answer": "a6"},
                {"question": "q7", "answer": "a7"},
                {"question": "q8", "answer": "a8"},
            ],
        },
    }
    total_questions = 8

    mock_identify_large.return_value = {}
    result = mock_hdbscan_clusterer._handle_large_clusters(
        clusters.copy(), total_questions
    )

    mock_identify_large.assert_called_once()
    assert mock_identify_large.call_args[0][0] == clusters
    assert mock_identify_large.call_args[0][1] == max(int(total_questions * 0.2), 50)

    mock_apply_clustering.assert_not_called()
    mock_create_subclusters.assert_not_called()

    assert result == clusters

    mock_identify_large.reset_mock()
    mock_apply_clustering.reset_mock()
    mock_create_subclusters.reset_mock()

    large_clusters = {"2": clusters["2"]}
    mock_identify_large.return_value = large_clusters

    mock_hdbscan_clusterer.embeddings_provider.get_embeddings = MagicMock()
    mock_embeddings = [np.array([0.1, 0.2]) for _ in range(5)]
    mock_hdbscan_clusterer.embeddings_provider.get_embeddings.return_value = (
        mock_embeddings
    )

    subcluster_labels = np.array([0, 0, 1, 1, 1])
    mock_apply_clustering.return_value = (subcluster_labels, 2)
    subclusters = {
        "2.0": {
            "questions": ["q4", "q5"],
            "qa_pairs": [
                {"question": "q4", "answer": "a4"},
                {"question": "q5", "answer": "a5"},
            ],
        },
        "2.1": {
            "questions": ["q6", "q7", "q8"],
            "qa_pairs": [
                {"question": "q6", "answer": "a6"},
                {"question": "q7", "answer": "a7"},
                {"question": "q8", "answer": "a8"},
            ],
        },
    }
    mock_create_subclusters.return_value = subclusters
    result = mock_hdbscan_clusterer._handle_large_clusters(
        clusters.copy(), total_questions
    )

    mock_identify_large.assert_called_once()
    mock_hdbscan_clusterer.embeddings_provider.get_embeddings.assert_called_once_with(
        clusters["2"]["questions"]
    )

    mock_apply_clustering.assert_called_once()
    assert mock_apply_clustering.call_args[0][0] == clusters["2"]["questions"]
    assert np.array_equal(
        mock_apply_clustering.call_args[0][1], np.array(mock_embeddings)
    )
    assert mock_apply_clustering.call_args[0][2] == "2"

    mock_create_subclusters.assert_called_once()
    assert mock_create_subclusters.call_args[0][0] == "2"
    assert mock_create_subclusters.call_args[0][1] == clusters["2"]["questions"]
    assert mock_create_subclusters.call_args[0][2] == clusters["2"]["qa_pairs"]
    assert np.array_equal(mock_create_subclusters.call_args[0][3], subcluster_labels)

    # Verify the result
    expected_result = {
        "1": clusters["1"],
        "2.0": subclusters["2.0"],
        "2.1": subclusters["2.1"],
    }
    assert result == expected_result


def test_custom_hdbscan_parameters():
    """Test that custom HDBSCAN parameters are used correctly."""
    with (
        patch("qadst.clusterer.HDBSCAN") as mock_hdbscan_class,
        patch("qadst.embeddings.get_embeddings_model"),
        patch("qadst.base.ChatOpenAI"),
    ):
        mock_hdbscan = MagicMock()
        mock_hdbscan.labels_ = np.array([0, 0, 1, 1, -1])
        mock_hdbscan.probabilities_ = np.array([0.9, 0.8, 0.7, 0.6, 0.0])
        mock_hdbscan_class.return_value = mock_hdbscan

        mock_embeddings_provider = MagicMock()
        mock_embeddings_provider.get_model_name.return_value = "test-model"

        clusterer = HDBSCANQAClusterer(
            embeddings_provider=mock_embeddings_provider,
            output_dir=tempfile.mkdtemp(),
            min_cluster_size=10,
            min_samples=5,
            cluster_selection_epsilon=0.5,
        )

        embeddings = np.random.rand(5, 10)
        clusterer.embeddings_provider.get_embeddings = MagicMock(
            return_value=embeddings
        )

        clusterer._perform_hdbscan_clustering(
            [("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3"), ("Q4", "A4"), ("Q5", "A5")]
        )

        mock_hdbscan_class.assert_called_once()
        _, kwargs = mock_hdbscan_class.call_args
        assert kwargs["min_cluster_size"] == 10
        assert kwargs["min_samples"] == 5
        assert kwargs["cluster_selection_epsilon"] == 0.5


def test_cluster_selection_method_parameter():
    """Test that the cluster_selection_method parameter is used correctly."""
    with (
        patch("qadst.clusterer.HDBSCAN") as mock_hdbscan_class,
        patch("qadst.embeddings.get_embeddings_model"),
        patch("qadst.base.ChatOpenAI"),
    ):
        # Create a mock HDBSCAN instance
        mock_hdbscan = MagicMock()
        mock_hdbscan.labels_ = np.array([0, 0, 1, 1, -1])
        mock_hdbscan.probabilities_ = np.array([0.9, 0.8, 0.7, 0.6, 0.0])
        mock_hdbscan_class.return_value = mock_hdbscan

        mock_embeddings_provider = MagicMock()
        mock_embeddings_provider.get_model_name.return_value = "test-model"

        clusterer = HDBSCANQAClusterer(
            embeddings_provider=mock_embeddings_provider,
            output_dir=tempfile.mkdtemp(),
            cluster_selection_method="leaf",
        )

        embeddings = np.random.rand(5, 10)
        clusterer.embeddings_provider.get_embeddings = MagicMock(
            return_value=embeddings
        )

        clusterer._perform_hdbscan_clustering(
            [("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3"), ("Q4", "A4"), ("Q5", "A5")]
        )

        mock_hdbscan_class.assert_called_once()
        _, kwargs = mock_hdbscan_class.call_args
        assert kwargs["cluster_selection_method"] == "leaf"


def test_keep_noise_parameter():
    """Test that the keep_noise parameter preserves noise points."""
    with (
        patch("qadst.clusterer.HDBSCAN"),
        patch("qadst.embeddings.get_embeddings_model"),
        patch("qadst.base.ChatOpenAI"),
    ):
        mock_embeddings_provider = MagicMock()
        mock_embeddings_provider.get_model_name.return_value = "test-model"

        clusterer = HDBSCANQAClusterer(
            embeddings_provider=mock_embeddings_provider,
            output_dir=tempfile.mkdtemp(),
            keep_noise=True,
        )

        clusters = {
            "0": {
                "questions": ["Q1", "Q2"],
                "qa_pairs": [
                    {"question": "Q1", "answer": "A1"},
                    {"question": "Q2", "answer": "A2"},
                ],
            },
            "-1": {
                "questions": ["Q3", "Q4"],
                "qa_pairs": [
                    {"question": "Q3", "answer": "A3"},
                    {"question": "Q4", "answer": "A4"},
                ],
            },
        }

        formatted = clusterer._format_clusters(clusters)

        # Find the noise cluster (should have id=0)
        noise_cluster = None
        for cluster in formatted["clusters"]:
            if cluster["id"] == 0:
                noise_cluster = cluster
                break

        assert noise_cluster is not None
        assert len(noise_cluster["representative"]) == 0  # No representative for noise
        assert len(noise_cluster["source"]) == 2
        assert noise_cluster["source"][0]["question"] in ["Q3", "Q4"]


def test_cluster_noise_points():
    """Test that the _cluster_noise_points method clusters noise points correctly."""
    with (
        patch("qadst.clusterer.HDBSCAN"),
        patch("qadst.embeddings.get_embeddings_model"),
        patch("qadst.base.ChatOpenAI"),
        patch("qadst.clusterer.KMeans"),
    ):
        mock_embeddings_provider = MagicMock()
        mock_embeddings_provider.get_model_name.return_value = "test-model"

        clusterer = HDBSCANQAClusterer(
            embeddings_provider=mock_embeddings_provider,
            output_dir=tempfile.mkdtemp(),
            keep_noise=False,
        )

        noise_qa_pairs = [
            ("Q1", "A1"),
            ("Q2", "A2"),
            ("Q3", "A3"),
        ]

        clusterer.embeddings_provider = MagicMock()
        clusterer.embeddings_provider.get_embeddings.return_value = [
            np.array([0.1, 0.2, 0.3]),
            np.array([0.4, 0.5, 0.6]),
            np.array([0.7, 0.8, 0.9]),
        ]

        kmeans_instance = MagicMock()
        kmeans_instance.fit_predict.return_value = np.array([0, 0, 1])
        with patch("qadst.clusterer.KMeans", return_value=kmeans_instance):
            result = clusterer._cluster_noise_points(noise_qa_pairs, min_cluster_size=2)

        assert len(result) == 2
        assert "0" in result
        assert "1" in result

        assert len(result["0"]["questions"]) == 2
        assert "Q1" in result["0"]["questions"]
        assert "Q2" in result["0"]["questions"]

        assert len(result["1"]["questions"]) == 1
        assert "Q3" in result["1"]["questions"]
