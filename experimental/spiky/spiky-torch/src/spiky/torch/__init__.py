# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""spiky.torch

Optional PyTorch integration.

Importing this module registers the PyTorch PrivateUse1 backend name as "nkipy".
NRT initialization remains explicit: call `spiky.init()` before allocating any
`torch.device("nkipy")` tensors.
"""

import os
import torch
import spiky as _spiky

from . import _C as _ext  # type: ignore
from .device import device_module


def _register():
    _ext._register_backend_name()
    _ext._register_hooks()
    _ext._register_allocator()

    # Make torch.nkipy available and generate standard methods (tensor.nkipy()).
    torch._register_device_module("nkipy", device_module)
    torch.utils.generate_methods_for_privateuse1_backend(
        for_tensor=True, for_module=True, for_storage=True
    )


_register()


def init_nkipy_backend(
    nkipy_cache: str | None = None,
    additional_compiler_args: str | None = None,
    keep_outputs_on_device: bool | None = None,
    pipelined: bool | None = None,
    device_id: int = 0,
) -> None:
    """Backward-compatible nanochat init hook.

    Newer spiky runtimes use process environment for compile/runtime knobs.
    Keep accepting legacy arguments so callers (e.g. nanochat) do not need to
    branch on spiky version.
    """
    if nkipy_cache is not None:
        os.environ["NANOCHAT_NKIPY_CACHE"] = nkipy_cache
    if additional_compiler_args is not None:
        os.environ["NANOCHAT_NKIPY_COMPILER_ARGS"] = additional_compiler_args
    if keep_outputs_on_device is not None:
        os.environ["NANOCHAT_NKIPY_KEEP_OUTPUTS_ON_DEVICE"] = (
            "1" if keep_outputs_on_device else "0"
        )
    if pipelined is not None:
        os.environ["NANOCHAT_NKIPY_PIPELINED"] = "1" if pipelined else "0"
    _spiky.init(int(device_id))


def is_nkipy_backend_initialized() -> bool:
    """Backward-compatible nanochat probe hook."""
    return bool(_spiky.is_initialized())


def device_count() -> int:
    return int(_ext._device_count())


def current_device() -> int:
    return int(_ext._current_device())


def set_device(device: int) -> None:
    _ext._set_device(int(device))


def is_available() -> bool:
    return bool(_ext._is_available())


def empty_cache() -> None:
    _ext._empty_cache()


def get_cached_blocks() -> int:
    return int(_ext._get_cached_blocks())


__all__ = [
    "init_nkipy_backend",
    "is_nkipy_backend_initialized",
    "device_count",
    "current_device",
    "set_device",
    "is_available",
    "empty_cache",
    "get_cached_blocks",
]
