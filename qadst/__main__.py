"""Entry point for the QA dataset clustering toolkit.

This module serves as the main entry point when the package is executed directly
using 'python -m qadst'. It initializes the command-line interface and passes
control to the main CLI function.
"""

import sys

from qadst.cli import main


def init() -> None:
    """Run qadst.cli.main() when current file is executed by an interpreter.

    If the file is used as a module, the qadst.cli.main() function will not
    automatically execute. The sys.exit() function is called with a return
    value of qadst.cli.main(), as all good UNIX programs do.
    """
    if __name__ == "__main__":
        sys.exit(main())


init()
