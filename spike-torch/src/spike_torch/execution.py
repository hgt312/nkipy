# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution utilities for spike-torch.

This module provides wrappers for loading and executing NRT models
using spike tensors.
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from spike_torch import _C


def nrt_init(rank: int = 0) -> None:
    """Initialize the Neuron Runtime.

    Args:
        rank: Process rank for distributed setup (used to set visible cores)
    """
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(rank)

    if os.environ.get("NEURON_RT_ROOT_COMM_ID") is None:
        root_addr = os.environ.get("MASTER_ADDR", "localhost")
        root_port = os.environ.get("NEURON_RT_PORT", "61234")
        os.environ["NEURON_RT_ROOT_COMM_ID"] = f"{root_addr}:{root_port}"

    _C._nrt_init()


def nrt_close() -> None:
    """Close the Neuron Runtime and clean up resources."""
    _C._nrt_close()


def nrt_load_model(
    neff_file: str | Path,
    cc_enabled: bool = False,
    device_id: int = 0,
    device_count: int = 1,
) -> int:
    """Load a NEFF model file.

    Args:
        neff_file: Path to the NEFF file
        cc_enabled: Whether to enable collective communication
        device_id: Device ID for collective communication
        device_count: Total device count for collective communication

    Returns:
        Model handle as an integer
    """
    neff_path = Path(neff_file)
    with open(neff_path, "rb") as f:
        neff_bytes = f.read()

    if cc_enabled:
        return _C._nrt_load_collectives(neff_bytes, device_id, device_count)
    else:
        return _C._nrt_load(neff_bytes)


def nrt_unload_model(model_handle: int) -> None:
    """Unload a model.

    Args:
        model_handle: Model handle returned by nrt_load_model
    """
    _C._nrt_unload(model_handle)


def nrt_execute_model(
    model_handle: int,
    inputs: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
) -> None:
    """Execute a model with given inputs and outputs.

    Args:
        model_handle: Model handle returned by nrt_load_model
        inputs: Dictionary mapping tensor names to input tensors (on spike device)
        outputs: Dictionary mapping tensor names to output tensors (on spike device)
    """
    # Allocate tensor sets
    input_set = _C._nrt_allocate_tensor_set()
    output_set = _C._nrt_allocate_tensor_set()

    try:
        # Add inputs
        for name, tensor in inputs.items():
            _C._nrt_add_tensor_to_tensor_set(input_set, tensor, name)

        # Add outputs
        for name, tensor in outputs.items():
            _C._nrt_add_tensor_to_tensor_set(output_set, tensor, name)

        # Execute
        _C._nrt_execute(model_handle, input_set, output_set)
    finally:
        # Always clean up tensor sets
        _C._nrt_destroy_tensor_set(input_set)
        _C._nrt_destroy_tensor_set(output_set)


def nrt_profile_start(model_handle: int, ntff_file: str | Path) -> None:
    """Start profiling for a model.

    Args:
        model_handle: Model handle returned by nrt_load_model
        ntff_file: Path to save the NTFF profile file
    """
    _C._nrt_profile_start(model_handle, str(ntff_file))


def nrt_profile_stop(ntff_file: str | Path) -> None:
    """Stop profiling and save results.

    Args:
        ntff_file: Path to the NTFF profile file
    """
    _C._nrt_profile_stop(str(ntff_file))


@contextmanager
def nrt_profile(model_handle: int, ntff_file: str | Path):
    """Context manager for profiling model execution.

    Args:
        model_handle: Model handle returned by nrt_load_model
        ntff_file: Path to save the NTFF profile file

    Example:
        with nrt_profile(model, "profile.ntff"):
            nrt_execute_model(model, inputs, outputs)
    """
    ntff_path = Path(ntff_file)
    ntff_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file
    if ntff_path.exists():
        ntff_path.unlink()

    nrt_profile_start(model_handle, str(ntff_path))
    try:
        yield
    finally:
        nrt_profile_stop(str(ntff_path))


def nrt_barrier(device_id: int, global_device_id: int, global_device_count: int) -> None:
    """Execute a barrier across all devices.

    Args:
        device_id: Local device ID
        global_device_id: Global device ID
        global_device_count: Total number of devices
    """
    _C._nrt_barrier(device_id, global_device_id, global_device_count)


__all__ = [
    "nrt_init",
    "nrt_close",
    "nrt_load_model",
    "nrt_unload_model",
    "nrt_execute_model",
    "nrt_profile_start",
    "nrt_profile_stop",
    "nrt_profile",
    "nrt_barrier",
]
