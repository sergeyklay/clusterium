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
    Convert a tensor to a numpy array.

    This function uses duck typing to detect PyTorch tensors by checking for
    the presence of the `detach()` method.

    Parameters
    ----------
    embedding : EmbeddingTensor
        The tensor to convert. Can be a PyTorch tensor or a numpy array.

    Returns
    -------
    numpy.ndarray
        The input converted to a numpy array. If the input is already a numpy array
        (or compatible), it is returned as is.
    """
    # Use duck typing to check if this is a PyTorch tensor
    # PyTorch tensors have detach() method, numpy arrays don't
    if hasattr(embedding, "detach"):
        return embedding.detach().cpu().numpy()  # type: ignore[attr-defined]

    # Already numpy or other array-like
    return np.asarray(embedding)
