"""
Entry point for direct module execution.

This module serves as the main entry point when the package is executed directly
using ``python -m clusx``.

It initializes the command-line interface and passes control to the main CLI function.

When executed with ``python -m clusx``, this module will initialize the CLI and
handle command-line arguments through the main function in the cli module.

See Also
--------
clusx.cli : Contains the main CLI implementation
"""

import sys

from clusx.cli import main


def init() -> None:
    """
    Run clusx.cli.main() when current file is executed by an interpreter.

    This function ensures that the CLI main function is only executed when this
    file is run directly, not when imported as a module.

    The :func:`sys.exit` function is called with the return value of
    :func:`clusx.cli.main`, following standard UNIX program conventions for exit
    codes.
    """
    if __name__ == "__main__":
        sys.exit(main())


init()
