# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

import torch

from torch_to_nkipy import _C
from torch_to_nkipy.backend.nkipy_backend_config import get_nkipy_backend_config
from torch_to_nkipy.device import device_module, distributed_backend  # noqa: F401
from torch_to_nkipy.utils.ntff_meta import NtffMeta

logger = logging.getLogger(__name__)

_C._register_nkipy_device()

# Register the PrivateUse1HooksInterface for nkipy
# This is required for torch.distributed operations (e.g., barrier) to work properly
_C._register_nkipy_hooks()

# Register the device module - this makes torch.nkipy available
torch._register_device_module("nkipy", device_module)

# Generate standard methods for the neuron backend
# This creates methods like tensor.neuron(), module.neuron(), etc.
torch.utils.generate_methods_for_privateuse1_backend(
    for_tensor=True, for_module=True, for_storage=True
)


def nrt_init(rank):
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(rank)

    if os.environ.get("NEURON_RT_ROOT_COMM_ID", None) is None:
        root_addr = os.environ.get("MASTER_ADDR", "localhost")
        root_port = os.environ.get("NEURON_RT_PORT", "61234")
        os.environ["NEURON_RT_ROOT_COMM_ID"] = f"{root_addr}:{root_port}"

    _C._nrt_init()


def nrt_close():
    _C._nrt_close()


def nrt_load_model(neff_file, cc_enabled, device_id, device_count):
    with open(neff_file, "rb") as f:
        neff_bytes = f.read()
    if cc_enabled:
        return _C._nrt_load_collectives(neff_bytes, device_id, device_count)
    else:
        return _C._nrt_load(neff_bytes)


def nrt_execute_model(model, inputs, outputs):
    # Create tensor sets
    input_set = _C._nrt_allocate_tensor_set()
    output_set = _C._nrt_allocate_tensor_set()

    for name, tensor in inputs.items():
        _C._nrt_add_tensor_to_tensor_set(input_set, tensor, name)

    # Add outputs
    for name, tensor in outputs.items():
        _C._nrt_add_tensor_to_tensor_set(output_set, tensor, name)

    # Execute
    _C._nrt_execute(model, input_set, output_set)

    # Destroy tensor sets
    _C._nrt_destroy_tensor_set(input_set)
    _C._nrt_destroy_tensor_set(output_set)


def nrt_profile_start(model, ntff_file):
    _C._nrt_profile_start(model, ntff_file)


def nrt_profile_stop(ntff_file):
    _C._nrt_profile_stop(ntff_file)


@contextmanager
def nrt_profile(model, ntff_meta: NtffMeta, neff_path: str):
    """Context manager controlling NRT profiling via NtffMeta."""
    logger.debug(
        f"Profiling NEFF with hash {ntff_meta.kernel_hash}, "
        f"save_ntff_exe_idx {ntff_meta.save_ntff_exe_idx}, and "
        f"curr_exe_idx {ntff_meta.curr_exe_idx}"
    )

    exe_idx_check = (
        not ntff_meta.save_ntff_exe_idx
        or ntff_meta.curr_exe_idx in ntff_meta.save_ntff_exe_idx
    )
    rank_check = get_nkipy_backend_config().rank == 0
    should_profile = ntff_meta.save_ntff and exe_idx_check and rank_check
    ntff_file = None

    if should_profile:
        save_dir = Path(
            f"{ntff_meta.save_ntff_dir}/kernel_{ntff_meta.kernel_hash}"
        ).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

        neff_filename = os.path.basename(neff_path)
        target_neff_path = save_dir / neff_filename
        if not target_neff_path.exists() and os.path.exists(neff_path):
            shutil.copy2(neff_path, target_neff_path)

        ntff_file = str(save_dir / f"{ntff_meta.curr_exe_idx}.ntff")
        if Path(ntff_file).exists():
            Path(ntff_file).unlink()

        logger.debug(f"Saving NTFF profile to {ntff_file}")
        nrt_profile_start(model=model, ntff_file=ntff_file)

    yield

    if should_profile:
        nrt_profile_stop(ntff_file=ntff_file)
    ntff_meta.curr_exe_idx += 1
