#!/usr/bin/env python3
"""
Simple script to run the clustering on a CSV file.
"""

import sys

from qa_dataset_clustering.cli import main
from qa_dataset_clustering.logging import setup_logging

if __name__ == "__main__":
    # Set up logging
    setup_logging()
    sys.exit(main())
