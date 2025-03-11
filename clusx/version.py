"""Version information.

This module provides version and metadata information for the package.
It attempts to get this information from the installed package metadata,
with fallback values for development environments.
"""

import importlib.metadata

try:
    _package_metadata = importlib.metadata.metadata("clusx")

    __version__ = _package_metadata.get("Version")
    __description__ = _package_metadata.get("Summary")
    __license__ = _package_metadata.get("License")
    __author__ = _package_metadata.get("Author")
    __author_email__ = _package_metadata.get("Author-email")
    __url__ = _package_metadata.get("Home-page")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0+dev"
    __description__ = "Clusterium"
    __license__ = "MIT"
    __author__ = "Serghei Iakovlev"
    __author_email__ = "oss@serghei.pl"
    __url__ = "https://github.com/sergeyklay/clusterium"

__copyright__ = f"Copyright (C) 2025 {__author__}"
