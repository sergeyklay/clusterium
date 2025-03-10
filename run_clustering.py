#!/usr/bin/env python3
"""
Simple script to run the clustering on a CSV file.
"""

import sys

from qadst.cli import main
from qadst.logging import setup_logging

if __name__ == "__main__":
    # Set up logging
    setup_logging()
    sys.exit(main())
