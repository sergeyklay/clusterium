"""Version information.

Provides package metadata through a cascading resolution strategy:
1. Installed package metadata (via importlib.metadata)
2. pyproject.toml (for development environments)
3. Fallback defaults
"""

import importlib
import importlib.metadata
import importlib.util
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional


@lru_cache(maxsize=1)
def get_metadata() -> Dict[str, str]:
    """Retrieve package metadata using a cascading resolution strategy."""
    resolvers = [
        _get_installed_metadata,
        _get_pyproject_metadata,
        _get_fallback_metadata,
    ]

    for resolver in resolvers:
        if metadata := resolver():
            return metadata

    return _get_fallback_metadata()  # Should never reach here, but just in case


def _get_installed_metadata() -> Optional[Dict[str, str]]:
    """Get metadata from installed package."""
    try:
        pkg_meta = importlib.metadata.metadata("clusx")
        return {
            "version": pkg_meta["Version"],
            "description": pkg_meta["Summary"],
            "license": pkg_meta["License"],
            "author": pkg_meta["Author"],
            "author_email": pkg_meta["Author-email"],
            "url": pkg_meta["Home-page"],
        }
    except (importlib.metadata.PackageNotFoundError, KeyError):
        return None


def _get_pyproject_metadata() -> Optional[Dict[str, str]]:
    """Get metadata from pyproject.toml."""
    pyproject_data = _find_and_parse_pyproject()
    if not pyproject_data:
        return None

    project = pyproject_data.get("project", {})
    poetry = pyproject_data.get("tool", {}).get("poetry", {})
    authors = project.get("authors", [])
    author_info = authors[0] if authors else {}

    return {
        "version": poetry.get("version", "0.0.0+dev"),
        "description": project.get("description", "Clusterium"),
        "license": project.get("license", "MIT"),
        "author": author_info.get("name", "Serghei Iakovlev"),
        "author_email": author_info.get("email", "oss@serghei.pl"),
        "url": project.get("urls", {}).get(
            "homepage", "https://clusterium.readthedocs.io"
        ),
    }


def _get_fallback_metadata() -> Dict[str, str]:
    """Get fallback metadata values."""
    return {
        "version": "0.0.0+dev",
        "description": "Clusterium",
        "license": "MIT",
        "author": "Serghei Iakovlev",
        "author_email": "oss@serghei.pl",
        "url": "https://clusterium.readthedocs.io",
    }


def _find_and_parse_pyproject() -> Optional[Dict[str, Any]]:
    """Find and parse pyproject.toml file."""
    # Traverse up from current directory to find pyproject.toml
    current_dir = Path(__file__).parent
    for _ in range(3):  # Check current and up to 2 parent directories
        pyproject_path = current_dir / "pyproject.toml"
        if pyproject_path.exists():
            return _parse_toml(pyproject_path)

        parent_dir = current_dir.parent
        if parent_dir == current_dir:  # Root directory reached
            break
        current_dir = parent_dir

    return None


def _parse_toml(path: Path) -> Optional[Dict[str, Any]]:
    """Parse TOML file with appropriate parser."""
    # Try stdlib tomllib (Python 3.11+) first, then fallback to tomli
    for module_name in ("tomllib", "tomli"):
        spec = importlib.util.find_spec(module_name)
        if not spec:
            continue

        try:
            toml_parser = importlib.import_module(module_name)
            with open(path, "rb") as f:
                return toml_parser.load(f)
        except Exception:
            continue

    return None


# Single source of truth
metadata = get_metadata()

# Export as module variables
__version__ = metadata["version"]
__description__ = metadata["description"]
__license__ = metadata["license"]
__author__ = metadata["author"]
__author_email__ = metadata["author_email"]
__url__ = metadata["url"]
__copyright__ = f"Copyright (C) 2025 {__author__}"
