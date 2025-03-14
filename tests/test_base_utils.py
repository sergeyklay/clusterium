"""Unit tests for the clustering utils module."""

import numpy as np
import torch

from clusx.utils import to_numpy


def test_pytorch_tensor_conversion():
    """Test converting a PyTorch tensor to numpy array."""
    torch_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    numpy_array = to_numpy(torch_tensor)

    assert isinstance(numpy_array, np.ndarray)
    assert np.array_equal(numpy_array, np.array([1.0, 2.0, 3.0, 4.0]))


def test_numpy_array_passthrough():
    """Test that numpy arrays are passed through correctly."""
    original_array = np.array([1.0, 2.0, 3.0, 4.0])
    result_array = to_numpy(original_array)

    assert isinstance(result_array, np.ndarray)
    assert np.array_equal(result_array, original_array)


def test_array_like_conversion():
    """Test converting an array-like object to numpy array."""
    array_like = np.array([1.0, 2.0, 3.0, 4.0])
    numpy_array = to_numpy(array_like)

    assert isinstance(numpy_array, np.ndarray)
    assert np.array_equal(numpy_array, np.array([1.0, 2.0, 3.0, 4.0]))


def test_multidimensional_conversion():
    """Test converting multidimensional data structures."""
    torch_tensor_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    numpy_array_2d = to_numpy(torch_tensor_2d)

    assert isinstance(numpy_array_2d, np.ndarray)
    assert numpy_array_2d.shape == (2, 2)
    assert np.array_equal(numpy_array_2d, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_empty_tensor_conversion():
    """Test converting an empty tensor."""
    empty_tensor = torch.tensor([])
    empty_array = to_numpy(empty_tensor)

    assert isinstance(empty_array, np.ndarray)
    assert empty_array.size == 0


def test_tensor_on_different_device():
    """Test converting a tensor that's on CPU (simulating different devices)."""
    cpu_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cpu")
    numpy_array = to_numpy(cpu_tensor)

    assert isinstance(numpy_array, np.ndarray)
    assert np.array_equal(numpy_array, np.array([1.0, 2.0, 3.0, 4.0]))
