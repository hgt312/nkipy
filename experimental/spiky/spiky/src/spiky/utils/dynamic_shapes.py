# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic shape discovery and bucket inference utilities."""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.fx.experimental.symbolic_shapes as symbolic_shapes
from torch.fx.experimental.symbolic_shapes import hint_int


def _safe_int(v):
    """Extract concrete int from SymInt without specializing the shape env."""
    if isinstance(v, torch.SymInt):
        return hint_int(v)
    return int(v)


@dataclass
class DynamicSpec:
    """Specification for a dynamic dimension.

    Attributes:
        arg_idx: Index of the input argument with dynamic dimension
        dim_idx: Index of the dynamic dimension within the tensor
        min_size: Minimum expected size (default: 1)
        max_size: Maximum expected size (default: 2048)
    """

    arg_idx: int
    dim_idx: int
    min_size: int = 1
    max_size: int = 2048


def discover_dynamic_specs(
    gm: Optional[Any],  # torch.fx.GraphModule, optional for flexibility
    example_inputs: Sequence,
) -> Dict[int, DynamicSpec]:
    """Discover dynamic dimensions from SymInt markers or dynamo attributes.

    Checks for:
    - SymInt in tensor shapes (from torch._dynamo.maybe_mark_dynamic)
    - tensor._dynamo_dynamic_indices attribute
    - tensor._dynamo_weak_dynamic_indices attribute

    Args:
        gm: FX GraphModule (currently unused, for future symbolic analysis)
        example_inputs: Sequence of example input tensors

    Returns:
        Dictionary mapping arg_idx to DynamicSpec for each dynamic input
    """
    specs: Dict[int, DynamicSpec] = {}
    debug = os.environ.get("SPIKY_DEBUG_DYNAMIC", "0") == "1"

    for arg_idx, inp in enumerate(example_inputs):
        if not hasattr(inp, "shape"):
            continue

        # Prefer explicit dynamo markers from maybe_mark_dynamic.
        dynamic_indices = getattr(inp, "_dynamo_dynamic_indices", None)
        if dynamic_indices:
            dim_idx = next(iter(dynamic_indices))
            specs[arg_idx] = DynamicSpec(
                arg_idx=arg_idx,
                dim_idx=dim_idx,
                min_size=1,
                max_size=_safe_int(inp.shape[dim_idx]) * 4,
            )
            if debug:
                print(
                    f"[dynamic-spec] arg={arg_idx} shape={tuple(inp.shape)} "
                    f"source=dynamic_indices dim={dim_idx}"
                )
            continue

        weak_indices = getattr(inp, "_dynamo_weak_dynamic_indices", None)
        if weak_indices:
            dim_idx = next(iter(weak_indices))
            specs[arg_idx] = DynamicSpec(
                arg_idx=arg_idx,
                dim_idx=dim_idx,
                min_size=1,
                max_size=_safe_int(inp.shape[dim_idx]) * 4,
            )
            if debug:
                print(
                    f"[dynamic-spec] arg={arg_idx} shape={tuple(inp.shape)} "
                    f"source=weak_dynamic_indices dim={dim_idx}"
                )
            continue

        # Next, ask symbolic-shape metadata for truly dynamic dims.
        # This avoids mistaking concrete SymInt-wrapped constants (e.g. head=2)
        # for dynamic dimensions.
        dyn_dims: List[int] = []
        try:
            for dim_idx in range(len(inp.shape)):
                if symbolic_shapes._is_dim_dynamic(inp, dim_idx):  # internal but stable in torch 2.8
                    dyn_dims.append(dim_idx)
        except Exception:
            dyn_dims = []

        if dyn_dims:
            dim_idx = min(dyn_dims)
            max_size = _safe_int(inp.shape[dim_idx]) * 4
            specs[arg_idx] = DynamicSpec(
                arg_idx=arg_idx,
                dim_idx=dim_idx,
                min_size=1,
                max_size=max_size,
            )
            if debug:
                print(
                    f"[dynamic-spec] arg={arg_idx} shape={tuple(inp.shape)} "
                    f"source=_is_dim_dynamic dims={dyn_dims} pick={dim_idx}"
                )
            continue

        # Fallback: infer from symbolic shape if marker attrs are unavailable.
        for dim_idx, dim in enumerate(inp.shape):
            if isinstance(dim, torch.SymInt) and not symbolic_shapes.is_concrete_int(dim):
                concrete_size = _safe_int(dim)
                specs[arg_idx] = DynamicSpec(
                    arg_idx=arg_idx,
                    dim_idx=dim_idx,
                    min_size=1,
                    max_size=concrete_size * 4,  # Allow 4x growth
                )
                if debug:
                    print(
                        f"[dynamic-spec] arg={arg_idx} shape={tuple(inp.shape)} "
                        f"source=symint dim={dim_idx}"
                    )
                break  # One dynamic dim per input

    # Graph-based fallback: backward graphs from AOT autograd may have
    # concrete example_inputs but symbolic FX placeholder metadata.
    if not specs and gm is not None:
        placeholder_idx = 0
        for node in gm.graph.nodes:
            if node.op != "placeholder":
                if node.op != "output":
                    continue
                break
            val = node.meta.get("val")
            if isinstance(val, torch.Tensor) and hasattr(val, "shape"):
                for dim_idx, dim in enumerate(val.shape):
                    if isinstance(dim, torch.SymInt):
                        specs[placeholder_idx] = DynamicSpec(
                            arg_idx=placeholder_idx,
                            dim_idx=dim_idx,
                            min_size=1,
                            max_size=max(_safe_int(dim) * 4, 64),
                        )
                        if debug:
                            print(
                                f"[dynamic-spec] arg={placeholder_idx} "
                                f"source=graph_placeholder dim={dim_idx}"
                            )
                        break  # One dynamic dim per input
            elif isinstance(val, torch.SymInt):
                # SymInt placeholder (dimension size passed as scalar)
                if debug:
                    print(
                        f"[dynamic-spec] skipping symint placeholder "
                        f"idx={placeholder_idx}"
                    )
            placeholder_idx += 1

    # Heuristic cleanup: drop low-capacity outlier specs when there is a
    # clear dominant dynamic extent (common in decomposed attention graphs
    # where tiny symbolic dims like num_heads can appear).
    if specs:
        max_cap = max(s.max_size for s in specs.values())
        # Only prune when the dominant cap is meaningfully larger.
        if max_cap >= 32:
            threshold = max_cap // 2
            filtered = {k: v for k, v in specs.items() if v.max_size >= threshold}
            if filtered and len(filtered) < len(specs):
                if debug:
                    dropped = [s for k, s in specs.items() if k not in filtered]
                    print(
                        "[dynamic-spec] dropped_outliers="
                        + ", ".join(
                            f"(arg={s.arg_idx},dim={s.dim_idx},max={s.max_size})"
                            for s in dropped
                        )
                    )
                specs = filtered

    if debug and specs:
        print(
            "[dynamic-spec] final="
            + ", ".join(
                f"(arg={s.arg_idx},dim={s.dim_idx},max={s.max_size})"
                for s in specs.values()
            )
        )

    # Some training graphs carry the same sequence length through multiple
    # integer tensors (e.g. token ids and targets), but only one gets explicit
    # dynamic markers. Include correlated integer tensors so padding/bucketing
    # stays consistent across both operands.
    if specs:
        primary_spec = max(
            specs.values(),
            key=lambda s: (s.max_size, -s.arg_idx, -s.dim_idx),
        )
        try:
            primary_inp = example_inputs[primary_spec.arg_idx]
            primary_extent = _safe_int(primary_inp.shape[primary_spec.dim_idx])
        except Exception:
            primary_extent = None
        if primary_extent is not None:
            for arg_idx, inp in enumerate(example_inputs):
                if arg_idx in specs or not isinstance(inp, torch.Tensor):
                    continue
                if inp.is_floating_point() or inp.dtype == torch.bool:
                    continue
                if len(inp.shape) <= primary_spec.dim_idx:
                    continue
                if _safe_int(inp.shape[primary_spec.dim_idx]) != primary_extent:
                    continue
                specs[arg_idx] = DynamicSpec(
                    arg_idx=arg_idx,
                    dim_idx=primary_spec.dim_idx,
                    min_size=1,
                    max_size=primary_spec.max_size,
                )
                if debug:
                    print(
                        f"[dynamic-spec] arg={arg_idx} shape={tuple(inp.shape)} "
                        f"source=correlated_int_extent dim={primary_spec.dim_idx}"
                    )

    return specs


def infer_buckets(
    dynamic_specs: Dict[int, DynamicSpec],
    min_size: int = 32,
    max_size: int = 2048,
    strategy: str = "power_of_2",
) -> List[int]:
    """Generate bucket sizes based on dynamic specs and strategy.

    Args:
        dynamic_specs: Dictionary of dynamic specifications
        min_size: Minimum bucket size
        max_size: Maximum bucket size
        strategy: Bucket generation strategy ("power_of_2" or "linear")

    Returns:
        Sorted list of bucket sizes
    """
    # Adjust max_size based on dynamic specs if available
    if dynamic_specs:
        spec_max = max(spec.max_size for spec in dynamic_specs.values())
        max_size = min(max_size, spec_max)

    if strategy == "power_of_2":
        buckets = []
        size = min_size
        while size <= max_size:
            buckets.append(size)
            size *= 2
        # Ensure we have at least one bucket
        if not buckets:
            buckets = [min_size]
        return buckets
    elif strategy == "linear":
        step = 64
        return list(range(min_size, max_size + 1, step))
    else:
        raise ValueError(f"Unknown bucket strategy: {strategy}")


def select_bucket(actual_len: int, buckets: List[int]) -> int:
    """Select the smallest bucket that fits the actual length.

    Args:
        actual_len: Actual sequence/dimension length
        buckets: List of available bucket sizes

    Returns:
        Selected bucket size (smallest bucket >= actual_len)

    Raises:
        ValueError: If actual_len exceeds all bucket sizes and no fallback
    """
    sorted_buckets = sorted(buckets)
    for bucket in sorted_buckets:
        if bucket >= actual_len:
            return bucket

    # Fallback to largest bucket (may need JIT compilation)
    return sorted_buckets[-1]
