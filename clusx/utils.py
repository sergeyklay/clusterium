"""
Utility functions for the clusx package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Union

    import torch
    from numpy.typing import NDArray

    EmbeddingTensor = Union[torch.Tensor, NDArray[np.float32]]


def to_numpy(embedding: EmbeddingTensor) -> NDArray[np.float32]:
    """
    A helper function to convert a tensor to a numpy array.

    If embedding is already a numpy array (or compatible), it is returned as is.
    Otherwise, it is converted to a numpy array.

    Args:
        embedding: The tensor to convert.

    Returns: The numpy array.
    """
    # Use duck typing to check if this is a PyTorch tensor
    # PyTorch tensors have detach() method, numpy arrays don't
    if hasattr(embedding, "detach"):
        return embedding.detach().cpu().numpy()  # type: ignore[attr-defined]

    # Already numpy or other array-like
    return np.asarray(embedding)
