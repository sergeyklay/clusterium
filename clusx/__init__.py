"""The top-level module for clusx.

This module tracks the version of the package as the base
package info used by various functions within the package.

Refer to the `documentation <https://clusterium.readthedocs.io/>`_ for
details on the use of this package.

"""

from .version import (
    __author__,
    __author_email__,
    __copyright__,
    __description__,
    __license__,
    __url__,
    __version__,
)

__all__ = [
    "__version__",
    "__description__",
    "__license__",
    "__author__",
    "__author_email__",
    "__url__",
    "__copyright__",
]
