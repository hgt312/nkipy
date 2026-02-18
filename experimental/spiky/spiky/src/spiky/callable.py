# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""NKIPyCallable - Dynamic shape callable with bucket selection and JIT compilation.

This module provides the bridge between torch-to-nkipy and spiky runtime:
- torch-to-nkipy provides: compiler_fn callback for bucket compilation
- spiky provides: bundle registration, bucket selection, padded execution
"""

import atexit
import logging
import threading
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

import spiky
from spiky.utils.dynamic_shapes import DynamicSpec
from spiky.utils.dynamic_shapes import select_bucket as _select_bucket
from spiky.utils.ntff_meta import NtffMeta

logger = logging.getLogger(__name__)

_KEEP_ON_DEVICE_WARNED = False

# dtype string (from C++ DeviceTensor) -> numpy dtype
_DTYPE_STR_TO_NUMPY = {
    "float32": np.float32,
    "float16": np.float16,
    "float64": np.float64,
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "uint32": np.uint32,
    "int64": np.int64,
    "uint64": np.uint64,
}

_TORCH_TO_NUMPY_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.int16: np.int16,
    torch.uint16: np.uint16,
    torch.int32: np.int32,
    torch.uint32: np.uint32,
    torch.int64: np.int64,
    torch.uint64: np.uint64,
    torch.bool: np.bool_,
}


def _normalize_numpy_int_buffer_format(arr: np.ndarray) -> np.ndarray:
    """Normalize numpy buffer formats for pybind type parsing.

    Some platforms expose NumPy int64/uint64 arrays with PEP3118 formats "l"/"L"
    instead of "q"/"Q". Our C++ buffer parser expects "q"/"Q".
    Also normalize bool buffers from "?" to uint8 ("B"), which has identical
    itemsize and byte representation for 0/1 values expected by runtime.
    """
    if not isinstance(arr, np.ndarray):
        return arr
    if arr.dtype == np.bool_:
        return arr.astype(np.uint8, copy=True)
    if arr.dtype == np.int64 and memoryview(arr).format == "l":
        return arr.astype(np.dtype("q"), copy=True)
    if arr.dtype == np.uint64 and memoryview(arr).format == "L":
        return arr.astype(np.dtype("Q"), copy=True)
    return arr


def _tensor_to_numpy_runtime(t: torch.Tensor) -> np.ndarray:
    """Convert tensor to NumPy for runtime execution buffer handoff."""
    if t.device.type != "cpu":
        t = t.detach().cpu().contiguous()
    else:
        t = t.detach().contiguous()

    if t.dtype == torch.bfloat16:
        # NumPy cannot directly export torch.bfloat16 buffers in this runtime.
        # Pass raw bf16 bits as uint16; runtime copies bytes by size.
        return _normalize_numpy_int_buffer_format(t.view(torch.uint16).numpy())

    return _normalize_numpy_int_buffer_format(t.numpy())


def _select_unpad_dim(
    shape: tuple,
    actual_len: Optional[int],
    dynamic_dim: Optional[int],
    padded_len: Optional[int],
) -> Optional[int]:
    """Pick a dimension to trim from padded to actual dynamic extent."""
    if actual_len is None:
        return None

    actual = int(actual_len)
    shape = tuple(int(d) for d in shape)

    if dynamic_dim is not None and 0 <= dynamic_dim < len(shape):
        # When padded_len is known, only unpad if the dimension actually
        # matches the padded (bucket) size.  This prevents incorrectly
        # truncating outputs whose shape at dynamic_dim is unrelated to
        # the padded batch extent (e.g. parameter gradients in backward
        # graphs where shape[0] is out_features, not the batch dim).
        if padded_len is not None:
            if shape[dynamic_dim] == int(padded_len):
                return dynamic_dim
            return None
        if shape[dynamic_dim] >= actual:
            return dynamic_dim
        return None

    candidates: List[int] = []
    if padded_len is not None:
        padded = int(padded_len)
        for i, dim in enumerate(shape):
            if dim == padded and dim >= actual:
                candidates.append(i)

    if not candidates:
        for i, dim in enumerate(shape):
            if dim > actual:
                candidates.append(i)

    if not candidates:
        return None

    for idx in candidates:
        if idx != 0:
            return idx
    return candidates[0]


def _maybe_unpad_array(
    arr: np.ndarray,
    actual_len: Optional[int],
    dynamic_dim: Optional[int],
    padded_len: Optional[int],
) -> np.ndarray:
    """Slice dynamic axis from padded extent down to actual_len when needed."""
    if actual_len is None:
        return arr

    unpad_dim = _select_unpad_dim(
        tuple(arr.shape),
        actual_len=actual_len,
        dynamic_dim=dynamic_dim,
        padded_len=padded_len,
    )
    if unpad_dim is None:
        return arr

    actual = int(actual_len)
    if arr.shape[unpad_dim] <= actual:
        return arr

    slices = [slice(None)] * arr.ndim
    slices[unpad_dim] = slice(0, actual)
    return arr[tuple(slices)]


def _decode_output_bytes_to_torch(
    raw_bytes: Any,
    expected_dtype: torch.dtype,
    expected_shape: tuple,
    device: torch.device,
    actual_len: Optional[int] = None,
    dynamic_dim: Optional[int] = None,
    padded_len: Optional[int] = None,
) -> torch.Tensor:
    """Decode raw output bytes into torch tensor using compile-time spec."""
    if isinstance(raw_bytes, memoryview):
        raw_bytes = raw_bytes.tobytes()
    elif not isinstance(raw_bytes, (bytes, bytearray)):
        # Some runtime paths surface a list[int] payload.
        raw_bytes = bytes(raw_bytes)

    shape = tuple(int(d) for d in expected_shape)
    numel = int(np.prod(shape, dtype=np.int64))
    unpad_dim = _select_unpad_dim(
        shape,
        actual_len=actual_len,
        dynamic_dim=dynamic_dim,
        padded_len=padded_len,
    )

    def _resolve_shape(itemsize: int) -> tuple:
        expected_bytes = numel * itemsize
        if len(raw_bytes) == expected_bytes:
            return shape
        if actual_len is not None and unpad_dim is not None:
            alt_shape = list(shape)
            alt_shape[unpad_dim] = int(actual_len)
            alt_shape = tuple(alt_shape)
            alt_numel = int(np.prod(alt_shape, dtype=np.int64))
            if len(raw_bytes) == alt_numel * itemsize:
                return alt_shape
        return shape

    if expected_dtype == torch.bool:
        # Some runtime paths materialize PRED outputs as 32-bit integers.
        bool_shape = _resolve_shape(np.dtype(np.bool_).itemsize)
        bool_numel = int(np.prod(bool_shape, dtype=np.int64))
        i32_shape = _resolve_shape(np.dtype(np.int32).itemsize)
        i32_numel = int(np.prod(i32_shape, dtype=np.int64))
        if len(raw_bytes) == bool_numel:
            arr = np.frombuffer(raw_bytes, dtype=np.bool_).reshape(bool_shape)
        elif len(raw_bytes) == i32_numel * np.dtype(np.int32).itemsize:
            arr = np.frombuffer(raw_bytes, dtype=np.int32).reshape(i32_shape) != 0
        else:
            raise ValueError(
                "Bool output byte-size mismatch: "
                f"expected={bool_numel} or {i32_numel * np.dtype(np.int32).itemsize}, got={len(raw_bytes)}"
            )
        arr = _maybe_unpad_array(
            arr,
            actual_len=actual_len,
            dynamic_dim=dynamic_dim,
            padded_len=padded_len,
        )
        return torch.from_numpy(arr.copy()).to(device)

    if expected_dtype == torch.bfloat16:
        bf16_shape = _resolve_shape(np.dtype(np.uint16).itemsize)
        arr = np.frombuffer(raw_bytes, dtype=np.uint16).reshape(bf16_shape)
        arr = _maybe_unpad_array(
            arr,
            actual_len=actual_len,
            dynamic_dim=dynamic_dim,
            padded_len=padded_len,
        )
        return torch.from_numpy(arr.copy()).view(torch.bfloat16).to(device)

    np_dtype = _TORCH_TO_NUMPY_DTYPE.get(expected_dtype)
    if np_dtype is None:
        raise ValueError(f"Unsupported output torch dtype: {expected_dtype}")
    typed_shape = _resolve_shape(np.dtype(np_dtype).itemsize)
    arr = np.frombuffer(raw_bytes, dtype=np_dtype).reshape(typed_shape)
    arr = _maybe_unpad_array(
        arr,
        actual_len=actual_len,
        dynamic_dim=dynamic_dim,
        padded_len=padded_len,
    )
    return torch.from_numpy(arr.copy()).to(device)


def _numpy_output_to_torch(
    output: np.ndarray,
    expected_dtype: torch.dtype,
    expected_shape: tuple,
    device: torch.device,
    actual_len: Optional[int] = None,
    dynamic_dim: Optional[int] = None,
    padded_len: Optional[int] = None,
) -> torch.Tensor:
    """Convert runtime numpy output to torch tensor using compile-time spec."""
    arr = np.asarray(output)
    shape = tuple(int(d) for d in expected_shape)

    if expected_dtype == torch.bfloat16:
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(np.uint16).itemsize
        if arr.nbytes != expected_bytes:
            raise ValueError(
                f"Output byte-size mismatch for bfloat16: expected={expected_bytes}, got={arr.nbytes}"
            )
        return _decode_output_bytes_to_torch(
            arr.tobytes(),
            expected_dtype,
            shape,
            device,
            actual_len=actual_len,
            dynamic_dim=dynamic_dim,
            padded_len=padded_len,
        )

    np_dtype = _TORCH_TO_NUMPY_DTYPE.get(expected_dtype)
    if np_dtype is None:
        raise ValueError(f"Unsupported output torch dtype: {expected_dtype}")

    if arr.dtype == np_dtype and tuple(arr.shape) == shape:
        arr = _maybe_unpad_array(
            arr,
            actual_len=actual_len,
            dynamic_dim=dynamic_dim,
            padded_len=padded_len,
        )
        return torch.from_numpy(arr.copy()).to(device)

    expected_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(np_dtype).itemsize
    if arr.nbytes != expected_bytes:
        alt_bytes = None
        unpad_dim = _select_unpad_dim(
            shape,
            actual_len=actual_len,
            dynamic_dim=dynamic_dim,
            padded_len=padded_len,
        )
        if actual_len is not None and unpad_dim is not None:
            alt_shape = list(shape)
            alt_shape[unpad_dim] = int(actual_len)
            alt_bytes = int(np.prod(tuple(alt_shape), dtype=np.int64)) * np.dtype(np_dtype).itemsize
        if alt_bytes is None or arr.nbytes != alt_bytes:
            raise ValueError(
                "Output byte-size mismatch after runtime execution: "
                f"expected={expected_bytes}, got={arr.nbytes}, "
                f"expected_shape={shape}, runtime_shape={tuple(arr.shape)}, "
                f"expected_dtype={expected_dtype}, runtime_dtype={arr.dtype}"
            )

    # Runtime tensor metadata may expose wrong dtype/shape ordering; reinterpret bytes
    # using compile-time IO specs to recover the intended tensor.
    return _decode_output_bytes_to_torch(
        arr.tobytes(),
        expected_dtype,
        shape,
        device,
        actual_len=actual_len,
        dynamic_dim=dynamic_dim,
        padded_len=padded_len,
    )


def _device_tensor_to_torch(
    device_tensor,
    device: torch.device,
    expected_dtype: Optional[torch.dtype] = None,
    expected_shape: Optional[tuple] = None,
    actual_len: Optional[int] = None,
    dynamic_dim: Optional[int] = None,
    padded_len: Optional[int] = None,
) -> torch.Tensor:
    """Convert a C++ DeviceTensor to a torch.Tensor via host-side copy."""
    raw_bytes = device_tensor.read_to_bytes()

    if expected_dtype is not None and expected_shape is not None:
        return _decode_output_bytes_to_torch(
            raw_bytes,
            expected_dtype,
            tuple(expected_shape),
            device,
            actual_len=actual_len,
            dynamic_dim=dynamic_dim,
            padded_len=padded_len,
        )

    np_dtype = _DTYPE_STR_TO_NUMPY.get(device_tensor.dtype)

    if np_dtype is None and device_tensor.dtype == "bfloat16":
        arr = np.frombuffer(bytes(raw_bytes), dtype=np.uint16).reshape(
            device_tensor.shape
        )
        arr = _maybe_unpad_array(
            arr,
            actual_len=actual_len,
            dynamic_dim=dynamic_dim,
            padded_len=padded_len,
        )
        return torch.from_numpy(arr.copy()).view(torch.bfloat16).to(device)

    if np_dtype is None:
        raise ValueError(f"Unsupported DeviceTensor dtype: {device_tensor.dtype}")

    arr = np.frombuffer(bytes(raw_bytes), dtype=np_dtype).reshape(device_tensor.shape)
    arr = _maybe_unpad_array(
        arr,
        actual_len=actual_len,
        dynamic_dim=dynamic_dim,
        padded_len=padded_len,
    )
    return torch.from_numpy(arr.copy()).to(device)


def _runtime_name_permutation(spec_names: List[str]) -> Optional[List[int]]:
    """Map runtime lexicographic order to compile-time order indices."""
    if not spec_names or len(set(spec_names)) != len(spec_names):
        return None
    compile_index_by_name = {name: idx for idx, name in enumerate(spec_names)}
    runtime_names = sorted(spec_names)
    return [compile_index_by_name[name] for name in runtime_names]


def _reorder_runtime_inputs(inputs: List[Any], input_specs: List[Any]) -> List[Any]:
    """Reorder compile-time input list to runtime lexicographic name order."""
    if not input_specs or len(inputs) != len(input_specs):
        return inputs
    compile_names = [spec.name for spec in input_specs]
    runtime_to_compile = _runtime_name_permutation(compile_names)
    if runtime_to_compile is None:
        return inputs
    return [inputs[compile_idx] for compile_idx in runtime_to_compile]


def _reorder_runtime_outputs(outputs: List[Any], output_specs: List[Any]) -> List[Any]:
    """Reorder runtime outputs to match compile-time output_specs order."""
    if not output_specs or len(outputs) != len(output_specs):
        return outputs

    expected_names = [spec.name for spec in output_specs]
    runtime_to_expected = _runtime_name_permutation(expected_names)
    if runtime_to_expected is None:
        return outputs

    reordered = [None] * len(outputs)
    for runtime_idx, expected_idx in enumerate(runtime_to_expected):
        reordered[expected_idx] = outputs[runtime_idx]
    if any(x is None for x in reordered):
        return outputs
    return reordered


# Track live callables for cleanup at interpreter shutdown.
# Using weak refs so callables can still be garbage-collected normally.
_live_callables: Dict[int, weakref.ref] = {}


def _cleanup_all_callables():
    """Flush and unregister all live callables before interpreter shutdown."""
    for ref in list(_live_callables.values()):
        obj = ref()
        if obj is not None:
            try:
                obj.close()
            except Exception:
                pass
    _live_callables.clear()


atexit.register(_cleanup_all_callables)


@dataclass
class CallableConfig:
    """Configuration for NKIPyCallable.

    Attributes:
        cache_dir: Directory for caching compiled NEFFs
        buckets: List of bucket sizes to use
        dynamic_specs: Dictionary mapping arg_idx to DynamicSpec
        jit_enabled: Whether to JIT compile new buckets on demand
        pipelined: Whether to use pipelined execution
        unpad_outputs: Whether to unpad outputs to original length
        cc_enabled: Whether collective communication is enabled
        rank_id: Process rank for collectives
        world_size: Total number of processes
        ntff_meta: Optional profiling metadata
    """

    cache_dir: Path
    buckets: List[int]
    dynamic_specs: Dict[int, DynamicSpec]
    symint_indices: List[int] = field(default_factory=list)
    jit_enabled: bool = True
    pipelined: bool = True
    unpad_outputs: bool = True
    pad_on_device: bool = True
    keep_outputs_on_device: bool = False
    input_layout: str = "auto"
    output_layout: str = "unpad"
    # Distributed
    cc_enabled: bool = False
    rank_id: int = 0
    world_size: int = 1
    # Profiling
    ntff_meta: Optional[NtffMeta] = None


class NKIPyCallable:
    """Callable wrapper handling bucket selection, JIT compilation, and padding.

    This class orchestrates dynamic shape execution:
    1. Analyzes input shapes to determine actual sequence length
    2. Selects appropriate bucket (or JIT compiles if needed)
    3. Registers bundle with spiky runtime
    4. Executes via spiky with device-side padding
    5. Returns outputs (optionally unpadded)

    Usage (from torch-to-nkipy):
        callable = NKIPyCallable(
            config=CallableConfig(...),
            compiler_fn=lambda bucket_size: compile_for_bucket(gm, bucket_size),
        )
        outputs = callable(*inputs)
    """

    def __init__(
        self,
        config: CallableConfig,
        compiler_fn: Callable[[int], Tuple[Any, ...]],
    ):
        """Initialize NKIPyCallable.

        Args:
            config: CallableConfig with bucket and execution settings
            compiler_fn: Callback to compile for a specific bucket size.
                         Signature: (bucket_size: int) ->
                         (neff_path, alias_map, non_tensor_outputs[, io_specs])
        """
        self._config = config
        self._compiler_fn = compiler_fn
        self._bundle_id: Optional[int] = None
        self._compiled_buckets: Dict[int, str] = {}  # bucket_size -> neff_path
        self._jit_lock = threading.Lock()
        self._buckets = list(config.buckets)  # Mutable copy
        self._alias_map: Optional[Dict[int, int]] = None
        # bucket_size -> {output_idx -> value}
        self._non_tensor_outputs: Dict[int, Dict[int, Any]] = {}
        # bucket_size -> IOSpecs
        self._io_specs_by_bucket: Dict[int, Any] = {}

        # Register for atexit cleanup
        _live_callables[id(self)] = weakref.ref(
            self, lambda ref, k=id(self): _live_callables.pop(k, None)
        )

    def _build_adjusted_dynamic_specs(self) -> Dict[int, int]:
        """Build dynamic_specs dict adjusted for SymInt arg removal.

        Returns arg_idx -> dim_idx mapping where arg_idx accounts for
        SymInt arguments that are filtered out before execute_bundle.
        """
        compile_to_runtime_idx = self._build_runtime_input_index_map()
        symint_set = set(self._config.symint_indices)
        dynamic_specs_dict = {}
        for spec in self._config.dynamic_specs.values():
            offset = sum(1 for si in symint_set if si < spec.arg_idx)
            compile_idx = spec.arg_idx - offset
            runtime_idx = compile_to_runtime_idx.get(compile_idx, compile_idx)
            dynamic_specs_dict[runtime_idx] = spec.dim_idx
        return dynamic_specs_dict

    def _build_runtime_input_index_map(self) -> Dict[int, int]:
        """Build compile-time input index -> runtime input index mapping."""
        for io_specs in self._io_specs_by_bucket.values():
            input_specs = getattr(io_specs, "input_specs", None)
            if not input_specs:
                continue
            compile_names = [spec.name for spec in input_specs]
            runtime_to_compile = _runtime_name_permutation(compile_names)
            if runtime_to_compile is None:
                return {}
            return {
                compile_idx: runtime_idx
                for runtime_idx, compile_idx in enumerate(runtime_to_compile)
            }
        return {}

    def _ensure_bundle_registered(self) -> None:
        """Register bundle with spiky if not already done."""
        if self._bundle_id is not None:
            return

        # Register with current compiled buckets (may be empty for JIT)
        self._bundle_id = spiky.register_bundle(
            bucket_to_neff=self._compiled_buckets.copy(),
            dynamic_specs=self._build_adjusted_dynamic_specs(),
            cc_enabled=self._config.cc_enabled,
            rank_id=self._config.rank_id,
            world_size=self._config.world_size,
        )

    def _ensure_bucket_compiled(self, bucket_size: int) -> None:
        """JIT compile a bucket if not already compiled."""
        if bucket_size in self._compiled_buckets:
            return

        with self._jit_lock:
            # Double-check after acquiring lock
            if bucket_size in self._compiled_buckets:
                return

            logger.info(f"JIT compiling bucket size {bucket_size}")
            compile_result = self._compiler_fn(bucket_size)
            if len(compile_result) == 4:
                neff_path, alias_map, non_tensor_outputs, io_specs = compile_result
            elif len(compile_result) == 3:
                neff_path, alias_map, non_tensor_outputs = compile_result
                io_specs = None
            else:
                raise ValueError(
                    "compiler_fn must return (neff_path, alias_map, non_tensor_outputs) "
                    "or (neff_path, alias_map, non_tensor_outputs, io_specs)"
                )
            self._compiled_buckets[bucket_size] = neff_path
            self._non_tensor_outputs[bucket_size] = non_tensor_outputs
            if io_specs is not None:
                self._io_specs_by_bucket[bucket_size] = io_specs
            if self._alias_map is None:
                self._alias_map = alias_map

            # Re-register bundle with updated buckets
            self._reregister_bundle()

    def _reregister_bundle(self) -> None:
        """Re-register bundle with current compiled buckets."""
        # Flush and unregister existing bundle if any
        if self._bundle_id is not None:
            try:
                spiky.flush_pipeline(self._bundle_id)
            except Exception:
                pass
            try:
                spiky.unregister_bundle(self._bundle_id)
            except Exception:
                pass  # Bundle may already be unregistered
            self._bundle_id = None

        self._bundle_id = spiky.register_bundle(
            bucket_to_neff=self._compiled_buckets.copy(),
            dynamic_specs=self._build_adjusted_dynamic_specs(),
            cc_enabled=self._config.cc_enabled,
            rank_id=self._config.rank_id,
            world_size=self._config.world_size,
        )

    def _determine_padding_strategy(self, args) -> Tuple[int, int, bool]:
        """Determine actual_len, bucket_size, and whether to skip padding.

        Implements input_layout logic:
        - "padded": requires pre-padded input with PaddingMetadata
        - "auto": detects pre-padded inputs, falls through to normal flow otherwise

        Args:
            args: Input tensors

        Returns:
            (actual_len, bucket_size, skip_padding)
        """
        from spiky.utils.tensor_metadata import get_metadata

        # Pick the most "capacity-driving" dynamic dim, not simply first-inserted.
        # Some graphs may contain tiny symbolic dims (e.g. num_heads=2) before
        # sequence-length symbols; selecting by max_size avoids bucketing on those.
        primary_spec = max(
            self._config.dynamic_specs.values(),
            key=lambda s: (s.max_size, -s.arg_idx, -s.dim_idx),
        )
        dyn_arg = args[primary_spec.arg_idx]
        meta = get_metadata(dyn_arg)

        if self._config.input_layout == "padded":
            # Require pre-padded input with metadata
            if meta is None:
                raise ValueError(
                    "input_layout='padded' requires input with PaddingMetadata"
                )
            if meta.pad_dim != primary_spec.dim_idx:
                raise ValueError(
                    f"Input padded on dim {meta.pad_dim}, "
                    f"expected {primary_spec.dim_idx}"
                )
            if meta.padded_size not in self._buckets:
                raise ValueError(
                    f"Input bucket {meta.padded_size} not in {self._buckets}"
                )
            return meta.original_size, meta.padded_size, True

        # input_layout == "auto"
        if meta is not None and meta.pad_dim == primary_spec.dim_idx:
            if meta.padded_size in self._buckets:
                return meta.original_size, meta.padded_size, True

        # Fall through to normal flow
        actual_len = dyn_arg.shape[primary_spec.dim_idx]
        bucket_size = _select_bucket(actual_len, self._buckets)
        return actual_len, bucket_size, False

    def __call__(self, *args: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Execute with automatic bucket selection and padding.

        Args:
            *args: Input tensors

        Returns:
            Tuple of output tensors
        """
        self._ensure_bundle_registered()

        # Handle static path (no dynamic specs) vs dynamic path
        if not self._config.dynamic_specs:
            bucket_size = self._buckets[0]
            actual_len = 0
            skip_padding = True
        else:
            # Determine padding strategy based on input_layout config
            actual_len, bucket_size, skip_padding = self._determine_padding_strategy(
                args
            )

        # Check if we need to JIT compile
        if bucket_size not in self._compiled_buckets:
            if self._config.jit_enabled:
                # Compile exact observed size for new buckets.
                # Power-of-2 up-rounding can over-stretch symbolic extents in
                # decomposed graphs (e.g. SDPA) and introduce shape mismatch.
                new_bucket = int(actual_len)
                if new_bucket not in self._buckets:
                    self._buckets = sorted(self._buckets + [new_bucket])
                bucket_size = new_bucket
            # else: will use the selected bucket even if not compiled (may fail)

        self._ensure_bucket_compiled(bucket_size)

        # Convert inputs to numpy for spiky.
        # Skip SymInt args (they were inlined as constants during compilation).
        symint_set = set(self._config.symint_indices)
        input_entries = []
        tensor_device = None
        for i, t in enumerate(args):
            if i in symint_set:
                continue
            if not isinstance(t, torch.Tensor):
                scalar_arr = np.array(t)
                input_entries.append(
                    {
                        "arr": _normalize_numpy_int_buffer_format(scalar_arr),
                        "is_scalar": True,
                        "arg_idx": i,
                    }
                )
                continue
            if tensor_device is None:
                tensor_device = t.device
            input_entries.append(
                {
                    "arr": _tensor_to_numpy_runtime(t),
                    "is_scalar": False,
                    "arg_idx": i,
                }
            )

        io_specs = self._io_specs_by_bucket.get(bucket_size)
        input_specs = io_specs.input_specs if io_specs is not None else None
        if input_specs and len(input_entries) != len(input_specs):
            expected = len(input_specs)
            actual = len(input_entries)
            if actual > expected:
                # Prefer dropping non-tensor scalar args first. Dynamic graphs can
                # still pass baked scalar extents (e.g., seq_len) at runtime even
                # when NEFF inputs contain tensors only.
                drop_budget = actual - expected
                trimmed_entries = []
                for entry in input_entries:
                    if drop_budget > 0 and entry["is_scalar"]:
                        drop_budget -= 1
                        continue
                    trimmed_entries.append(entry)
                input_entries = trimmed_entries
                if len(input_entries) == expected:
                    logger.debug(
                        "Dropped %d scalar runtime arg(s) to match NEFF input specs (%d).",
                        actual - expected,
                        expected,
                    )
            # Do not attempt to synthesize missing inputs. Keep current entries and
            # let runtime validation fail with clear shape/count diagnostics.
        inputs_np = [entry["arr"] for entry in input_entries]
        if input_specs:
            inputs_np = _reorder_runtime_inputs(inputs_np, input_specs)

        # When inputs are already padded, skip device-side padding
        pad_on_device = self._config.pad_on_device and not skip_padding

        # Wrap execution in optional profiling context
        ntff_meta = self._config.ntff_meta
        if (
            ntff_meta is not None
            and ntff_meta.save_ntff
            and bucket_size in self._compiled_buckets
        ):
            from spiky.device.profiling import nkipy_profile

            profile_ctx = nkipy_profile(ntff_meta, self._compiled_buckets[bucket_size])
        else:
            from contextlib import nullcontext

            profile_ctx = nullcontext((False, None))

        with profile_ctx as (save_trace, ntff_file):
            # Execute via spiky (handles padding internally on device)
            if self._config.pipelined:
                outputs = spiky.execute_pipelined(
                    bundle_id=self._bundle_id,
                    bucket_size=bucket_size,
                    inputs=inputs_np,
                    # No self-prefetch: content-blind hit check causes
                    # stale data.
                    next_inputs=[],
                    pad_on_device=pad_on_device,
                    keep_outputs_on_device=self._config.keep_outputs_on_device,
                    unpad_outputs=self._config.unpad_outputs,
                    actual_len=actual_len if self._config.unpad_outputs else 0,
                    save_trace=save_trace,
                    ntff_name=ntff_file or "",
                )
            else:
                outputs = spiky.execute_bundle(
                    bundle_id=self._bundle_id,
                    bucket_size=bucket_size,
                    inputs=inputs_np,
                    pad_on_device=pad_on_device,
                    keep_outputs_on_device=self._config.keep_outputs_on_device,
                    unpad_outputs=self._config.unpad_outputs,
                    actual_len=actual_len if self._config.unpad_outputs else 0,
                    save_trace=save_trace,
                    ntff_name=ntff_file or "",
                )

        output_specs = io_specs.output_specs if io_specs is not None else None
        if output_specs:
            outputs = _reorder_runtime_outputs(list(outputs), output_specs)

        # Convert back to torch tensors
        device = tensor_device or torch.device("cpu")
        primary_dynamic_dim = None
        if self._config.dynamic_specs:
            primary_spec = max(
                self._config.dynamic_specs.values(),
                key=lambda s: (s.max_size, -s.arg_idx, -s.dim_idx),
            )
            primary_dynamic_dim = primary_spec.dim_idx
        if self._config.keep_outputs_on_device:
            global _KEEP_ON_DEVICE_WARNED
            if not _KEEP_ON_DEVICE_WARNED:
                logger.warning(
                    "keep_outputs_on_device currently performs a host roundtrip "
                    "for tensor conversion. True zero-copy device tensors require "
                    "spike-torch integration (not yet implemented)."
                )
                _KEEP_ON_DEVICE_WARNED = True
            result = []
            for i, out in enumerate(outputs):
                if output_specs and i < len(output_specs):
                    spec = output_specs[i]
                    result.append(
                        _device_tensor_to_torch(
                            out,
                            device,
                            expected_dtype=spec.dtype,
                            expected_shape=spec.shape,
                            actual_len=actual_len if self._config.unpad_outputs else None,
                            dynamic_dim=primary_dynamic_dim,
                            padded_len=bucket_size if self._config.unpad_outputs else None,
                        )
                    )
                else:
                    result.append(
                        _device_tensor_to_torch(
                            out,
                            device,
                            actual_len=actual_len if self._config.unpad_outputs else None,
                            dynamic_dim=primary_dynamic_dim,
                            padded_len=bucket_size if self._config.unpad_outputs else None,
                        )
                    )
        else:
            result = []
            for i, out in enumerate(outputs):
                if output_specs and i < len(output_specs):
                    spec = output_specs[i]
                    result.append(
                        _numpy_output_to_torch(
                            out,
                            expected_dtype=spec.dtype,
                            expected_shape=spec.shape,
                            device=device,
                            actual_len=actual_len if self._config.unpad_outputs else None,
                            dynamic_dim=primary_dynamic_dim,
                            padded_len=bucket_size if self._config.unpad_outputs else None,
                        )
                    )
                else:
                    arr = np.asarray(out)
                    arr = _maybe_unpad_array(
                        arr,
                        actual_len=actual_len if self._config.unpad_outputs else None,
                        dynamic_dim=primary_dynamic_dim,
                        padded_len=bucket_size if self._config.unpad_outputs else None,
                    )
                    result.append(torch.from_numpy(arr.copy()).to(device))

        # Attach metadata when output_layout="padded"
        if self._config.output_layout == "padded" and self._config.dynamic_specs:
            from spiky.utils.tensor_metadata import PaddingMetadata, attach_metadata

            primary_spec = list(self._config.dynamic_specs.values())[0]
            for out in result:
                attach_metadata(
                    out,
                    PaddingMetadata(
                        original_size=actual_len,
                        padded_size=bucket_size,
                        pad_dim=primary_spec.dim_idx,
                        arg_indices=tuple(self._config.dynamic_specs.keys()),
                    ),
                )

        # Handle alias_map: copy aliased outputs back to input tensors
        if self._alias_map:
            for output_idx, input_idx in self._alias_map.items():
                if output_idx < len(result):
                    args[input_idx].copy_(result[output_idx].to(args[input_idx].device))
            result = [r for i, r in enumerate(result) if i not in self._alias_map]

        # Handle non-tensor outputs: insert actual values at specified positions
        non_tensor_map = self._non_tensor_outputs.get(bucket_size, {})
        if non_tensor_map:
            for idx in sorted(non_tensor_map.keys()):
                result.insert(idx, non_tensor_map[idx])

        return tuple(result)

    def flush(self) -> None:
        """Flush any pending pipelined execution."""
        if self._bundle_id is not None:
            spiky.flush_pipeline(self._bundle_id)

    def close(self) -> None:
        """Unregister bundle and clean up resources."""
        _live_callables.pop(id(self), None)
        if self._bundle_id is not None:
            try:
                self.flush()
            except Exception:
                pass
            try:
                spiky.unregister_bundle(self._bundle_id)
            except Exception:
                pass
            self._bundle_id = None
        self._compiled_buckets.clear()
        self._io_specs_by_bucket.clear()

    @property
    def buckets(self) -> List[int]:
        """Get current list of bucket sizes."""
        return self._buckets.copy()

    @property
    def compiled_buckets(self) -> Dict[int, str]:
        """Get dictionary of compiled buckets (bucket_size -> neff_path)."""
        return self._compiled_buckets.copy()

    def __del__(self):
        """Auto-flush on garbage collection."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during interpreter shutdown
