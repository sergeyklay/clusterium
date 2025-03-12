"""Unit tests for the clustering utils module."""

import csv
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from clusx.clustering.utils import (
    get_embeddings,
    load_cluster_assignments,
    load_data,
    save_clusters_to_csv,
    save_clusters_to_json,
)


def test_load_data_from_csv_basic(basic_qa_csv):
    """Test basic functionality of load_data with CSV file."""
    texts = load_data(str(basic_qa_csv), column="question")

    assert len(texts) == 2
    assert texts[0] == "What is Python?"
    assert texts[1] == "What is TensorFlow?"


def test_load_data_from_csv_custom_columns(custom_columns_csv):
    """Test load_data with custom column names."""
    texts = load_data(str(custom_columns_csv), column="query")

    assert len(texts) == 2
    assert texts[0] == "What is Python?"
    assert texts[1] == "What is TensorFlow?"


def test_load_data_from_csv_empty_rows(csv_with_empty_rows):
    """Test load_data with empty rows in CSV."""
    texts = load_data(str(csv_with_empty_rows), column="question")

    assert len(texts) == 2
    assert texts[0] == "What is Python?"
    assert texts[1] == "What is TensorFlow?"


def test_load_data_from_text_file(basic_text_file):
    """Test load_data with a text file."""
    texts = load_data(str(basic_text_file))

    assert len(texts) == 3
    assert texts[0] == "What is Python?"
    assert texts[1] == "What is TensorFlow?"
    assert texts[2] == "What is PyTorch?"


def test_load_data_csv_without_column(basic_qa_csv):
    """Test load_data with CSV file but without specifying a column."""
    with pytest.raises(ValueError, match="Column name must be specified"):
        load_data(str(basic_qa_csv))


def test_save_clusters_to_csv(tmp_path, sample_texts, sample_clusters):
    """Test basic functionality of save_clusters_to_csv."""
    output_path = tmp_path / "clusters.csv"

    save_clusters_to_csv(str(output_path), sample_texts, sample_clusters, "DP")

    assert output_path.exists()

    with open(output_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    assert header == ["Text", "Cluster_DP", "Alpha", "Sigma", "Variance"]
    assert len(rows) == 3
    assert rows[0][0] == "What is Python?"
    assert rows[0][1] == "0"
    assert rows[0][2] == "1.0"  # Default alpha value
    assert rows[0][3] == "0.0"  # Default sigma value
    assert rows[0][4] == "0.1"  # Default variance value
    assert rows[1][0] == "What is TensorFlow?"
    assert rows[1][1] == "1"
    assert rows[2][0] == "What is PyTorch?"
    assert rows[2][1] == "1"


def test_save_clusters_to_json(tmp_path, sample_texts, sample_clusters):
    """Test basic functionality of save_clusters_to_json."""
    output_path = tmp_path / "clusters.json"

    save_clusters_to_json(str(output_path), sample_texts, sample_clusters, "DP")

    assert output_path.exists()

    with open(output_path, "r") as f:
        data = json.load(f)

    assert "clusters" in data
    assert len(data["clusters"]) == 2

    cluster1 = data["clusters"][0]
    assert cluster1["id"] == 1
    assert cluster1["representative"] == "What is Python?"
    assert "members" in cluster1
    assert len(cluster1["members"]) == 1
    assert cluster1["members"][0] == "What is Python?"

    cluster2 = data["clusters"][1]
    assert cluster2["id"] == 2
    assert cluster2["representative"] == "What is TensorFlow?"
    assert "members" in cluster2
    assert len(cluster2["members"]) == 2
    assert cluster2["members"][0] == "What is TensorFlow?"
    assert cluster2["members"][1] == "What is PyTorch?"

    assert "metadata" in data
    assert data["metadata"]["model_name"] == "DP"
    assert data["metadata"]["alpha"] == 1.0
    assert data["metadata"]["sigma"] == 0.0
    assert data["metadata"]["variance"] == 0.1


@patch("clusx.clustering.DirichletProcess")
def test_get_embeddings(mock_dp_class):
    """Test get_embeddings with mocked DirichletProcess."""
    mock_dp = MagicMock()
    mock_dp_class.return_value = mock_dp

    mock_embedding = np.array([0.1, 0.2, 0.3])
    mock_dp.get_embedding.return_value = mock_embedding

    texts = ["What is Python?", "What is TensorFlow?"]

    embeddings = get_embeddings(texts)

    assert mock_dp_class.called
    assert mock_dp_class.call_args[1]["alpha"] == 1.0
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
