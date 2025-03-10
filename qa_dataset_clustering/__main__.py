"""
Entry point for direct module execution.
"""

import sys

from qa_dataset_clustering.cli import main
from qa_dataset_clustering.logging import setup_logging

if __name__ == "__main__":
    # Set up logging
    setup_logging()
    sys.exit(main())
