# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from torch_to_nkipy import _C

current_device_idx = 0


def current_device() -> int:
    """Return the current Neuron device index."""

    global current_device_idx
    return current_device_idx


def set_device(device: int) -> None:
    """Set the current Neuron device.

    Args:
        device: Device index to set as current
    """

    global current_device_idx
    current_device_idx = device


def device_count() -> int:
    """Return the number of Neuron devices available."""

    return _C._nrt_device_count()


def get_amp_supported_dtype():
    """Return list of dtypes supported by Neuron for automatic mixed precision."""
    import torch

    return [torch.float32, torch.float16, torch.bfloat16]
