"""
Unit tests for the visualization module.
"""

from clusx.visualization import MIN_DATASET_SIZE, is_small_dataset


def test_empty_reports():
    """Test with empty reports dictionary.

    An empty dataset should be considered small since it contains zero data points,
    which is less than any positive threshold.
    """
    assert is_small_dataset({}, MIN_DATASET_SIZE)


def test_no_cluster_stats():
    """Test when reports don't contain cluster_stats.

    If no reports have cluster_stats, the dataset should be considered small.
    """
    reports = {
        "model1": {"metrics": {"some_metric": 0.5}},
        "model2": {"parameters": {"alpha": 0.1}},
    }
    assert is_small_dataset(reports, MIN_DATASET_SIZE)


def test_no_num_texts():
    """Test when cluster_stats doesn't contain num_texts.

    If no reports have num_texts in their cluster_stats, the dataset should be
    considered small.
    """
    reports = {
        "model1": {"cluster_stats": {"cluster_sizes": {}}},
        "model2": {"cluster_stats": {"num_clusters": 5}},
    }
    assert is_small_dataset(reports, MIN_DATASET_SIZE)


def test_all_datasets_large_enough():
    """Test when all datasets have enough texts."""
    reports = {
        "model1": {"cluster_stats": {"num_texts": MIN_DATASET_SIZE}},
        "model2": {"cluster_stats": {"num_texts": MIN_DATASET_SIZE + 5}},
    }
    assert not is_small_dataset(reports, MIN_DATASET_SIZE)


def test_one_dataset_too_small():
    """Test when one dataset has too few texts."""
    reports = {
        "model1": {"cluster_stats": {"num_texts": MIN_DATASET_SIZE}},
        "model2": {"cluster_stats": {"num_texts": MIN_DATASET_SIZE - 1}},
    }
    assert is_small_dataset(reports, MIN_DATASET_SIZE)


def test_all_datasets_too_small():
    """Test when all datasets have too few texts."""
    reports = {
        "model1": {"cluster_stats": {"num_texts": MIN_DATASET_SIZE - 1}},
        "model2": {"cluster_stats": {"num_texts": MIN_DATASET_SIZE - 2}},
    }
    assert is_small_dataset(reports, MIN_DATASET_SIZE)


def test_mixed_report_structure():
    """Test with mixed report structure (some with num_texts, some without).

    As long as at least one report has num_texts and it's large enough,
    the dataset should not be considered small.
    """
    reports = {
        "model1": {"cluster_stats": {"num_texts": MIN_DATASET_SIZE + 5}},
        "model2": {"cluster_stats": {"num_clusters": 10}},  # No num_texts
        "model3": {"metrics": {"some_metric": 0.5}},  # No cluster_stats
    }
    assert not is_small_dataset(reports, MIN_DATASET_SIZE)


def test_mixed_report_structure_all_small():
    """Test with mixed report structure where all reports with num_texts are small."""
    reports = {
        "model1": {"cluster_stats": {"num_texts": MIN_DATASET_SIZE - 1}},
        "model2": {"cluster_stats": {"num_clusters": 10}},  # No num_texts
        "model3": {"metrics": {"some_metric": 0.5}},  # No cluster_stats
    }
    assert is_small_dataset(reports, MIN_DATASET_SIZE)


def test_custom_threshold():
    """Test with a custom threshold value."""
    custom_threshold = 20
    reports = {
        "model1": {"cluster_stats": {"num_texts": 15}},
        "model2": {"cluster_stats": {"num_texts": 25}},
    }
    # First model is below custom threshold
    assert is_small_dataset(reports, custom_threshold)


def test_zero_threshold():
    """Test with a threshold of zero."""
    reports = {
        "model1": {"cluster_stats": {"num_texts": 0}},
        "model2": {"cluster_stats": {"num_texts": 1}},
    }
    # No dataset should be considered small with threshold 0
    assert not is_small_dataset(reports, 0)


def test_negative_num_texts():
    """Test with negative num_texts values (edge case)."""
    reports = {
        "model1": {"cluster_stats": {"num_texts": -5}},
        "model2": {"cluster_stats": {"num_texts": MIN_DATASET_SIZE}},
    }
    # Negative values are less than any positive threshold
    assert is_small_dataset(reports, MIN_DATASET_SIZE)
