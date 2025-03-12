"""Common test fixtures, mocks and configurations."""

import csv

import pytest


@pytest.fixture
def basic_qa_csv(tmp_path):
    """Create a basic CSV file with question-answer pairs."""
    csv_path = tmp_path / "test.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer"])
        writer.writerow(["What is Python?", "Python is a programming language."])
        writer.writerow(
            ["What is TensorFlow?", "TensorFlow is a machine learning framework."]
        )
    return csv_path


@pytest.fixture
def custom_columns_csv(tmp_path):
    """Create a CSV file with custom column names."""
    csv_path = tmp_path / "test_custom.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "response"])
        writer.writerow(["What is Python?", "Python is a programming language."])
        writer.writerow(
            ["What is TensorFlow?", "TensorFlow is a machine learning framework."]
        )
    return csv_path


@pytest.fixture
def csv_with_empty_rows(tmp_path):
    """Create a CSV file with empty rows."""
    csv_path = tmp_path / "test_empty.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer"])
        writer.writerow(["What is Python?", "Python is a programming language."])
        writer.writerow(["", "This row should be skipped."])
        writer.writerow(
            ["What is TensorFlow?", "TensorFlow is a machine learning framework."]
        )
    return csv_path


@pytest.fixture
def basic_text_file(tmp_path):
    """Create a basic text file with one text per line."""
    text_path = tmp_path / "test.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("What is Python?\n")
        f.write("What is TensorFlow?\n")
        f.write("\n")  # Empty line should be skipped
        f.write("What is PyTorch?\n")
    return text_path


@pytest.fixture
def sample_texts():
    """Return a list of sample question texts."""
    return ["What is Python?", "What is TensorFlow?", "What is PyTorch?"]


@pytest.fixture
def sample_clusters():
    """Return a list of sample cluster assignments."""
    return [0, 1, 1]


@pytest.fixture
def sample_data():
    """Return a list of sample data dictionaries."""
    return [
        {
            "question": "What is Python?",
            "answer": "Python is a programming language.",
        },
        {
            "question": "What is TensorFlow?",
            "answer": "TensorFlow is a machine learning framework.",
        },
        {
            "question": "What is PyTorch?",
            "answer": "PyTorch is another machine learning framework.",
        },
    ]


@pytest.fixture
def cluster_assignments_csv(tmp_path):
    """Create a CSV file with cluster assignments."""
    csv_path = tmp_path / "cluster_assignments.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Text", "Cluster_DP"])
        writer.writerow(["What is Python?", "0"])
        writer.writerow(["What is TensorFlow?", "1"])
        writer.writerow(["What is PyTorch?", "1"])
    return csv_path


@pytest.fixture
def cluster_assignments_custom_column_csv(tmp_path):
    """Create a CSV file with cluster assignments using a custom column name."""
    csv_path = tmp_path / "cluster_assignments_custom.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Text", "ClusterCustom"])
        writer.writerow(["What is Python?", "0"])
        writer.writerow(["What is TensorFlow?", "1"])
        writer.writerow(["What is PyTorch?", "1"])
    return csv_path


@pytest.fixture
def cluster_assignments_no_cluster_column_csv(tmp_path):
    """Create a CSV file without a cluster column."""
    csv_path = tmp_path / "no_cluster_column.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Text", "OtherColumn"])
        writer.writerow(["What is Python?", "value1"])
        writer.writerow(["What is TensorFlow?", "value2"])
        writer.writerow(["What is PyTorch?", "value3"])
    return csv_path
