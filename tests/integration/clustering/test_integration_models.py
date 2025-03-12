"""Integration tests for the DirichletProcess and PitmanYorProcess classes."""

import pytest

from clusx.clustering.models import DirichletProcess, PitmanYorProcess


@pytest.fixture
def sample_texts() -> list[str]:
    """Return a list of sample texts for clustering."""
    return [
        "the sky is so high",
        "the sky is blue",
        "fly high into the sky.",
        "the trees are really tall",
        "I love the trees",
        "trees make me happy",
        "the sun is shining really bright",
    ]


@pytest.mark.integration
def test_fit_with_real_data_dp(sample_texts: list[str]):
    """Test fitting the model with real data."""
    dp = DirichletProcess(alpha=1.0)
    clusters, _ = dp.fit(sample_texts)

    assert len(clusters) == len(sample_texts)


@pytest.mark.integration
def test_fit_with_real_data_pyp(sample_texts: list[str]):
    """Test fitting the model with real data."""
    pyp = PitmanYorProcess(alpha=1.0, sigma=0.5)
    clusters, _ = pyp.fit(sample_texts)

    assert len(clusters) == len(sample_texts)


@pytest.mark.integration
def test_compare_pyp_dp(sample_texts: list[str]):
    """Compare PitmanYorProcess with DirichletProcess."""
    dp = DirichletProcess(alpha=1.0)
    pyp = PitmanYorProcess(alpha=1.0, sigma=0.5)

    dp_clusters, _ = dp.fit(sample_texts)
    pyp_clusters, _ = pyp.fit(sample_texts)

    assert len(dp_clusters) == len(pyp_clusters)

    dp_unique_clusters = len(set(dp_clusters))
    pyp_unique_clusters = len(set(pyp_clusters))

    assert dp_unique_clusters > 0
    assert pyp_unique_clusters > 0
