# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""spike-torch: PyTorch device integration for spike runtime on AWS Neuron.

This module provides PyTorch device registration using spike as the runtime.
It enables:
- torch.device("nkipy") for creating tensors on Neuron devices
- tensor.to("nkipy") for moving tensors to/from Neuron devices
- model.nkipy() for moving modules to Neuron devices
"""

import torch

from spike_torch import _C
from spike_torch.device import device_module

# Step 1: Initialize NRT runtime
_C._nrt_init()

# Step 2: Register 'nkipy' as the name for PrivateUse1 device type (C++ side)
_C._register_spike_device()

# Step 3: Register the PrivateUse1HooksInterface for autograd support
_C._register_spike_hooks()

# Step 4: Register the device module - this makes torch.nkipy available
torch._register_device_module("nkipy", device_module)

# Step 5: Generate standard methods for the nkipy backend
# This creates methods like tensor.spike(), module.spike(), etc.
torch.utils.generate_methods_for_privateuse1_backend(
    for_tensor=True, for_module=True, for_storage=True
)


def device_count() -> int:
    """Return the number of available spike devices."""
    return _C._device_count()


def current_device() -> int:
    """Return the current spike device index."""
    return _C._current_device()


def set_device(device: int) -> None:
    """Set the current spike device."""
    _C._set_device(device)


def is_available() -> bool:
    """Return True if spike devices are available."""
    return _C._is_available()


def empty_cache() -> None:
    """Release all unused cached memory from the spike memory pool."""
    _C._empty_cache()


def get_cached_blocks() -> int:
    """Return the number of cached memory blocks."""
    return _C._get_cached_blocks()


def get_nrt_tensor(tensor: torch.Tensor) -> int | None:
    """Get the underlying NRT tensor handle for a spike tensor.

    Args:
        tensor: A PyTorch tensor on spike device

    Returns:
        The NRT tensor handle as an integer, or None if not found
    """
    return _C._get_nrt_tensor(tensor)


__all__ = [
    "device_count",
    "current_device",
    "set_device",
    "is_available",
    "empty_cache",
    "get_cached_blocks",
    "get_nrt_tensor",
]
