"""Unit tests for the clustering utils module."""

import csv
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qadst.clustering.utils import (
    get_embeddings,
    load_cluster_assignments,
    load_data_from_csv,
    save_clusters_to_csv,
    save_clusters_to_json,
)


def test_load_data_from_csv_basic(basic_qa_csv):
    """Test basic functionality of load_data_from_csv."""
    texts, data = load_data_from_csv(str(basic_qa_csv))

    assert len(texts) == 2
    assert texts[0] == "What is Python?"
    assert texts[1] == "What is TensorFlow?"
    assert len(data) == 2
    assert data[0]["question"] == "What is Python?"
    assert data[0]["answer"] == "Python is a programming language."
    assert data[1]["question"] == "What is TensorFlow?"
    assert data[1]["answer"] == "TensorFlow is a machine learning framework."


def test_load_data_from_csv_custom_columns(custom_columns_csv):
    """Test load_data_from_csv with custom column names."""
    texts, data = load_data_from_csv(
        str(custom_columns_csv), column="query", answer_column="response"
    )

    assert len(texts) == 2
    assert texts[0] == "What is Python?"
    assert texts[1] == "What is TensorFlow?"
    assert len(data) == 2
    assert data[0]["query"] == "What is Python?"
    assert data[0]["response"] == "Python is a programming language."
    assert data[1]["query"] == "What is TensorFlow?"
    assert data[1]["response"] == "TensorFlow is a machine learning framework."


def test_load_data_from_csv_empty_rows(csv_with_empty_rows):
    """Test load_data_from_csv with empty rows."""
    texts, data = load_data_from_csv(str(csv_with_empty_rows))

    assert len(texts) == 2
    assert texts[0] == "What is Python?"
    assert texts[1] == "What is TensorFlow?"
    assert len(data) == 2


def test_save_clusters_to_csv(tmp_path, sample_texts, sample_clusters):
    """Test basic functionality of save_clusters_to_csv."""
    output_path = tmp_path / "clusters.csv"

    save_clusters_to_csv(str(output_path), sample_texts, sample_clusters, "DP")

    assert output_path.exists()

    with open(output_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    assert header == ["Text", "Cluster_DP", "Alpha", "Sigma"]
    assert len(rows) == 3
    assert rows[0][0] == "What is Python?"
    assert rows[0][1] == "0"
    assert rows[0][2] == "1.0"  # Default alpha value
    assert rows[0][3] == "0.0"  # Default sigma value
    assert rows[1][0] == "What is TensorFlow?"
    assert rows[1][1] == "1"
    assert rows[2][0] == "What is PyTorch?"
    assert rows[2][1] == "1"


def test_save_clusters_to_json_basic(tmp_path, sample_texts, sample_clusters):
    """Test basic functionality of save_clusters_to_json without data."""
    output_path = tmp_path / "clusters.json"

    save_clusters_to_json(str(output_path), sample_texts, sample_clusters, "DP")

    assert output_path.exists()

    with open(output_path, "r") as f:
        data = json.load(f)

    assert "clusters" in data
    assert len(data["clusters"]) == 2  # Two unique clusters

    cluster1 = data["clusters"][0]
    assert cluster1["id"] == 1
    assert cluster1["representative"][0]["question"] == "What is Python?"
    assert "Answer for cluster 1" in cluster1["representative"][0]["answer"]
    assert len(cluster1["source"]) == 1

    cluster2 = data["clusters"][1]
    assert cluster2["id"] == 2
    assert cluster2["representative"][0]["question"] == "What is TensorFlow?"
    assert "Answer for cluster 2" in cluster2["representative"][0]["answer"]
    assert len(cluster2["source"]) == 2
    assert cluster2["source"][0]["question"] == "What is TensorFlow?"
    assert cluster2["source"][1]["question"] == "What is PyTorch?"


def test_save_clusters_to_json_with_data(
    tmp_path, sample_texts, sample_clusters, sample_data
):
    """Test save_clusters_to_json with data containing answers."""
    output_path = tmp_path / "clusters_with_data.json"

    save_clusters_to_json(
        str(output_path), sample_texts, sample_clusters, "DP", sample_data
    )

    assert output_path.exists()

    with open(output_path, "r") as f:
        result = json.load(f)

    assert "clusters" in result
    assert len(result["clusters"]) == 2  # Two unique clusters

    cluster1 = result["clusters"][0]
    assert cluster1["id"] == 1
    assert cluster1["representative"][0]["question"] == "What is Python?"
    assert (
        cluster1["representative"][0]["answer"] == "Python is a programming language."
    )
    assert len(cluster1["source"]) == 1

    cluster2 = result["clusters"][1]
    assert cluster2["id"] == 2
    assert cluster2["representative"][0]["question"] == "What is TensorFlow?"
    assert (
        cluster2["representative"][0]["answer"]
        == "TensorFlow is a machine learning framework."
    )
    assert len(cluster2["source"]) == 2
    assert cluster2["source"][0]["question"] == "What is TensorFlow?"
    assert (
        cluster2["source"][0]["answer"] == "TensorFlow is a machine learning framework."
    )
    assert cluster2["source"][1]["question"] == "What is PyTorch?"
    assert (
        cluster2["source"][1]["answer"]
        == "PyTorch is another machine learning framework."
    )


@patch("qadst.clustering.DirichletProcess")
def test_get_embeddings(mock_dp_class):
    """Test get_embeddings with mocked DirichletProcess."""
    mock_dp = MagicMock()
    mock_dp_class.return_value = mock_dp

    mock_embedding = np.array([0.1, 0.2, 0.3])
    mock_dp.get_embedding.return_value = mock_embedding

    mock_cache = MagicMock()
    texts = ["What is Python?", "What is TensorFlow?"]

    embeddings = get_embeddings(texts, mock_cache)

    assert mock_dp_class.called
    assert mock_dp_class.call_args[1]["alpha"] == 1.0
    assert mock_dp_class.call_args[1]["cache"] == mock_cache
    assert mock_dp.get_embedding.call_count == 2
    assert len(embeddings) == 2
    assert np.array_equal(embeddings[0], mock_embedding)
    assert np.array_equal(embeddings[1], mock_embedding)


def test_load_cluster_assignments(cluster_assignments_csv):
    """Test basic functionality of load_cluster_assignments."""
    clusters, params = load_cluster_assignments(str(cluster_assignments_csv))

    assert len(clusters) == 3
    assert clusters == [0, 1, 1]
    assert isinstance(params, dict)
    assert "alpha" in params
    assert "sigma" in params
    assert params["alpha"] == 1.0  # Default value
    assert params["sigma"] == 0.0  # Default value


def test_load_cluster_assignments_custom_column(cluster_assignments_custom_column_csv):
    """Test load_cluster_assignments with a custom cluster column name."""
    clusters, params = load_cluster_assignments(
        str(cluster_assignments_custom_column_csv)
    )

    assert len(clusters) == 3
    assert clusters == [0, 1, 1]
    assert isinstance(params, dict)
    assert "alpha" in params
    assert "sigma" in params


def test_load_cluster_assignments_no_cluster_column(
    cluster_assignments_no_cluster_column_csv,
):
    """Test load_cluster_assignments with a non-existent cluster column."""
    with pytest.raises(ValueError):
        load_cluster_assignments(str(cluster_assignments_no_cluster_column_csv))
