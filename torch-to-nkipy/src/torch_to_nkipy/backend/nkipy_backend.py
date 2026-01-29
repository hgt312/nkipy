# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NKIPy backend implementation for PyTorch Dynamo."""

import builtins
import logging
import os
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import torch
import torch._dynamo
import torch._functorch._aot_autograd.runtime_wrappers as runtime_wrappers
import torch.fx as fx
from torch._decomp import core_aten_decompositions
from torch._dynamo.backends.registry import register_backend
from torch._functorch._aot_autograd.utils import make_boxed_func
from torch._functorch.aot_autograd import aot_module_simplified
from torch._inductor.utils import InputType
from torch_to_nkipy.backend.nkipy_backend_config import (
    NKIPyBackendConfig,
    get_nkipy_backend_config,
    reset_nkipy_backend_config,
    set_nkipy_backend_config,
)
from torch_to_nkipy.device import nrt_close, nrt_init
from torch_to_nkipy.nkipy_builder.nkipy_kernel import NKIPyKernel
from torch_to_nkipy.utils.graph import _count_subgraph_markers as count_subgraph_markers
from torch_to_nkipy.utils.graph import gm_split_and_wrap

# FIXME replace the AliasOfInputHandler in aot_module_simplified
runtime_wrappers.AliasOfInputHandler.__call__ = (
    runtime_wrappers.NoopAliasHandler.__call__
)

logger = logging.getLogger(__name__)


def is_nkipy_backend_initialized() -> bool:
    """Check if NKIPy backend is initialized."""
    return get_nkipy_backend_config() is not None


def init_nkipy_backend(
    nkipy_cache: str = "./nkipy_cache",
    log_level: int = logging.INFO,
    rank: int = 0,
    world_size: int = 1,
    additional_compiler_args: str = "",
) -> None:
    """Initialize the NKIPy backend.

    Args:
        nkipy_cache: Directory path for cache storage
        log_level: Logging level
        rank: Process rank for distributed setup
        world_size: Total number of processes
        additional_compiler_args: Custom compiler flags

    Raises:
        RuntimeError: If backend is already initialized
    """
    if is_nkipy_backend_initialized():
        raise RuntimeError("NKIPy backend has already been initialized.")

    # FIXME Currently each process (TP rank) should have its own nkipy_cache
    nkipy_cache = str(Path(nkipy_cache).resolve())
    os.makedirs(nkipy_cache, exist_ok=True)

    logging.basicConfig(level=log_level)

    nrt_init(rank)

    nkipy_backend_config = NKIPyBackendConfig(
        nkipy_cache_prefix=nkipy_cache,
        log_level=log_level,
        rank=rank,
        world_size=world_size,
        additional_compiler_args=additional_compiler_args,
    )
    set_nkipy_backend_config(nkipy_backend_config)

    logger.debug(f"NKIPy backend initialized with config: {nkipy_backend_config}")


def reset_nkipy_backend():
    reset_nkipy_backend_config()
    nrt_close()


class CompiledWrapper(torch.nn.Module):
    def __init__(
        self,
        gm: fx.GraphModule,
        options: Optional[dict[str, Union[str, builtins.int, builtins.bool]]] = None,
    ):
        super().__init__()
        self._gm = gm
        self._handle = None
        self._options = options

    def forward(self, *args, **kwargs):
        if self._handle is None:
            self._handle = NKIPyKernel(self._gm, args, self._options)
        return self._handle(*args, **kwargs)


def nkipy_backend_fn_decomposed(
    gm: fx.GraphModule,
    example_inputs: Sequence[InputType],
    options: Optional[dict[str, Union[str, builtins.int, builtins.bool]]] = None,
) -> Callable:
    """Decompose the graph for NKIPy backend.

    Args:
        graph: The FX graph module to decompose
        example_inputs: Example inputs for the graph

    Returns:
        Callable: The decomposed graph
    """

    if count_subgraph_markers(gm) == 0:
        compiled_fn = CompiledWrapper(gm, options)
    else:
        compiled_fn = gm_split_and_wrap(gm, CompiledWrapper, options)

    return make_boxed_func(compiled_fn)


def nkipy_backend_fn(
    gm: fx.GraphModule,
    example_inputs: Sequence[InputType],
    options: Optional[dict[str, Union[str, builtins.int, builtins.bool]]] = None,
) -> Callable:
    """Main backend function for NKIPy.

    Args:
        graph: The FX graph module to compile
        example_inputs: Example inputs for the graph

    Returns:
        Callable: The compiled function
    """
    # FIXME Kernel-centric Execution is not supported yet
    # FIXME We are still using torch's aot_module_simplified

    def fw_compiler_with_options(gm, example_inputs):
        return nkipy_backend_fn_decomposed(gm, example_inputs, options=options)

    return aot_module_simplified(
        gm,
        example_inputs,
        fw_compiler=fw_compiler_with_options,
        decompositions=core_aten_decompositions(),
        keep_inference_input_mutations=True,
    )


register_backend(name="nkipy", compiler_fn=nkipy_backend_fn)
