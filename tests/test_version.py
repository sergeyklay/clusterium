"""Tests for the version module."""

import importlib.metadata
import importlib.util
from pathlib import Path
from unittest import mock

from clusx.version import (
    __author__,
    __author_email__,
    __copyright__,
    __description__,
    __license__,
    __url__,
    __version__,
    _find_and_parse_pyproject,
    _get_fallback_metadata,
    _get_installed_metadata,
    _get_pyproject_metadata,
    _parse_toml,
    get_metadata,
)


def test_version_exports():
    """Test that all version exports are strings and not empty."""
    for var in [
        __version__,
        __description__,
        __license__,
        __author__,
        __author_email__,
        __url__,
        __copyright__,
    ]:
        assert isinstance(var, str)
        assert var  # not empty


def test_get_metadata_caching():
    """Test that get_metadata caches its result."""
    result1 = get_metadata()
    result2 = get_metadata()
    assert result1 is result2


@mock.patch("clusx.version._get_installed_metadata")
@mock.patch("clusx.version._get_pyproject_metadata")
@mock.patch("clusx.version._get_fallback_metadata")
def test_get_metadata_cascade(mock_fallback, mock_pyproject, mock_installed):
    """Test the cascading resolution strategy."""
    get_metadata.cache_clear()

    mock_installed.return_value = {"version": "1.0.0", "description": "Installed"}
    mock_pyproject.return_value = {"version": "2.0.0", "description": "Pyproject"}
    mock_fallback.return_value = {"version": "3.0.0", "description": "Fallback"}

    result = get_metadata()
    assert result["version"] == "1.0.0"
    assert result["description"] == "Installed"
    mock_installed.assert_called_once()
    mock_pyproject.assert_not_called()
    mock_fallback.assert_not_called()

    mock_installed.reset_mock()
    mock_pyproject.reset_mock()
    mock_fallback.reset_mock()
    get_metadata.cache_clear()

    mock_installed.return_value = None
    mock_pyproject.return_value = {"version": "2.0.0", "description": "Pyproject"}
    mock_fallback.return_value = {"version": "3.0.0", "description": "Fallback"}

    result = get_metadata()
    assert result["version"] == "2.0.0"
    assert result["description"] == "Pyproject"
    mock_installed.assert_called_once()
    mock_pyproject.assert_called_once()
    mock_fallback.assert_not_called()

    mock_installed.reset_mock()
    mock_pyproject.reset_mock()
    mock_fallback.reset_mock()
    get_metadata.cache_clear()

    mock_installed.return_value = None
    mock_pyproject.return_value = None
    mock_fallback.return_value = {"version": "3.0.0", "description": "Fallback"}

    result = get_metadata()
    assert result["version"] == "3.0.0"
    assert result["description"] == "Fallback"
    mock_installed.assert_called_once()
    mock_pyproject.assert_called_once()
    mock_fallback.assert_called_once()


def test_get_fallback_metadata():
    """Test the fallback metadata values."""
    fallback = _get_fallback_metadata()
    assert fallback["version"] == "0.0.0+dev"
    assert fallback["description"] == "Clusterium"
    assert fallback["license"] == "MIT"
    assert fallback["author"] == "Serghei Iakovlev"
    assert fallback["author_email"] == "oss@serghei.pl"
    assert fallback["url"] == "https://clusterium.readthedocs.io"


@mock.patch("importlib.metadata.metadata")
def test_get_installed_metadata_found(mock_metadata):
    """Test getting metadata from an installed package."""
    mock_metadata_dict = {
        "Version": "1.2.3",
        "Summary": "Test Description",
        "License": "Test License",
        "Author": "Test Author",
        "Author-email": "test@example.com",
        "Home-page": "https://example.com",
    }
    mock_metadata.return_value.get = lambda key, default=None: mock_metadata_dict.get(
        key, default
    )

    result = _get_installed_metadata()
    assert result is not None
    assert result["version"] == "1.2.3"
    assert result["description"] == "Test Description"
    assert result["license"] == "Test License"
    assert result["author"] == "Test Author"
    assert result["author_email"] == "test@example.com"
    assert result["url"] == "https://example.com"


@mock.patch(
    "importlib.metadata.metadata", side_effect=importlib.metadata.PackageNotFoundError
)
def test_get_installed_metadata_not_found(mock_metadata):
    """Test behavior when package is not installed."""
    result = _get_installed_metadata()
    assert result is None


@mock.patch("clusx.version._find_and_parse_pyproject")
def test_get_pyproject_metadata_found(mock_find_parse):
    """Test getting metadata from pyproject.toml."""
    mock_find_parse.return_value = {
        "project": {
            "description": "Project Description",
            "license": "Project License",
            "authors": [{"name": "Project Author", "email": "project@example.com"}],
            "urls": {"homepage": "https://project.example.com"},
        },
        "tool": {
            "poetry": {
                "version": "2.3.4",
            }
        },
    }

    result = _get_pyproject_metadata()
    assert result is not None
    assert result["version"] == "2.3.4"
    assert result["description"] == "Project Description"
    assert result["license"] == "Project License"
    assert result["author"] == "Project Author"
    assert result["author_email"] == "project@example.com"
    assert result["url"] == "https://project.example.com"


@mock.patch("clusx.version._find_and_parse_pyproject", return_value=None)
def test_get_pyproject_metadata_not_found(mock_find_parse):
    """Test behavior when pyproject.toml is not found."""
    result = _get_pyproject_metadata()
    assert result is None


@mock.patch("clusx.version._parse_toml")
def test_find_and_parse_pyproject(mock_parse_toml):
    """Test finding and parsing pyproject.toml."""
    mock_parse_toml.return_value = {"test": "data"}

    with mock.patch.object(Path, "exists", return_value=True):
        result = _find_and_parse_pyproject()
        assert result == {"test": "data"}
        mock_parse_toml.assert_called_once()


@mock.patch("clusx.version._parse_toml")
def test_find_and_parse_pyproject_not_found(mock_parse_toml):
    """Test behavior when pyproject.toml is not found in any parent directory."""
    with mock.patch.object(Path, "exists", return_value=False):
        result = _find_and_parse_pyproject()
        assert result is None
        mock_parse_toml.assert_not_called()


@mock.patch("importlib.util.find_spec")
def test_parse_toml_no_parser(mock_find_spec):
    """Test behavior when no TOML parser is available."""
    mock_find_spec.return_value = None

    result = _parse_toml(Path("dummy/path"))
    assert result is None


@mock.patch("importlib.util.find_spec")
def test_parse_toml_exception(mock_find_spec):
    """Test behavior when TOML parsing raises an exception."""
    mock_find_spec.return_value = mock.MagicMock()

    with mock.patch("clusx.version.importlib.import_module") as mock_import:
        mock_import.side_effect = Exception("Test exception")

        result = _parse_toml(Path("dummy/path"))
        assert result is None


@mock.patch("clusx.version._parse_toml")
def test_find_and_parse_pyproject_traversal(mock_parse_toml):
    """Test that _find_and_parse_pyproject traverses up directories."""
    mock_parse_toml.return_value = {"test": "data"}

    original_exists = Path.exists
    call_count = 0

    def mock_exists(self):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return False
        elif call_count == 2:
            return True
        return original_exists(self)

    with mock.patch.object(Path, "exists", mock_exists):
        result = _find_and_parse_pyproject()
        assert result == {"test": "data"}
        assert call_count == 2
        assert mock_parse_toml.call_count == 1


@mock.patch("clusx.version._parse_toml")
def test_find_and_parse_pyproject_max_traversal(mock_parse_toml):
    """Test that _find_and_parse_pyproject stops after max traversals."""
    mock_parse_toml.reset_mock()

    mock_parse_toml.return_value = None

    with mock.patch.object(Path, "exists", return_value=False):
        with mock.patch.object(Path, "parent", return_value=Path("/mock/root")):
            result = _find_and_parse_pyproject()
            assert result is None


def test_module_variables():
    """Test that all module variables are exported correctly."""
    import clusx.version as version_module

    assert hasattr(version_module, "__version__")
    assert hasattr(version_module, "__description__")
    assert hasattr(version_module, "__license__")
    assert hasattr(version_module, "__author__")
    assert hasattr(version_module, "__author_email__")
    assert hasattr(version_module, "__url__")
    assert hasattr(version_module, "__copyright__")

    assert version_module.__author__ in version_module.__copyright__
