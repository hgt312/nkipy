# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast

import torch.fx as fx

from torch_to_nkipy.nkipy_builder.aten_op_registry.base import AtenOpRegistry
from torch_to_nkipy.nkipy_builder.nkipy_ast import ComputationNode
from torch_to_nkipy.utils.graph import get_shape_from_fx_node


@AtenOpRegistry.register("torch.ops.aten.index_put.default")
def index_put_default(node: fx.Node, computation_node: ComputationNode) -> None:
    """
    Handle index_put operation.

    Maps PyTorch's index_put to NumPy advanced indexing operations:
    index_put = np.copy(x)
    index_put[index] = value

    Args:
        node: The FX node representing the index_put operation
        computation_node: The ComputationNode to add code to
    """
    ast_block = computation_node.ast_code_block

    # Extract arguments
    source = node.args[0]
    indices = node.args[1]
    value = node.args[2]

    # accumulate=True requires scatter-add semantics (e.g., embedding backward).
    accumulate = False
    if len(node.args) > 3:
        if isinstance(node.args[3], bool):
            accumulate = node.args[3]
        elif isinstance(node.args[3], fx.Node):
            raise NotImplementedError("index_put with non-constant accumulate is not supported")
        else:
            accumulate = bool(node.args[3])
    elif "accumulate" in node.kwargs:
        accumulate = bool(node.kwargs["accumulate"])

    # Create index string from the indices list
    index_strs = []
    for index in indices:
        if index is None:
            index_strs.append("")
        elif isinstance(index, fx.Node):
            index_strs.append(index.name)
        else:
            raise TypeError(
                f"Unexpected index type: {type(index)}. Expected None or fx.Node."
            )

    # Pad with empty strings for any remaining dimensions
    target_dim = len(get_shape_from_fx_node(node))
    index_strs.extend([""] * (target_dim - len(index_strs)))

    # Step 1: Create a copy of the source tensor
    ast_block.add_numpy_call_assignment(
        target=node.name, func_name="copy", args=[source.name]
    )

    # Step 2: Apply indexed update
    if accumulate:
        # For index_put(..., accumulate=True), emit np.add.at(dst, index, value)
        # instead of plain indexed assignment to preserve scatter-add semantics.
        if index_strs and index_strs[0] and all(s == "" for s in index_strs[1:]):
            # Common embedding-backward pattern: add along leading axis.
            index_expr = index_strs[0]
        else:
            tuple_parts = ["slice(None)" if s == "" else s for s in index_strs]
            index_expr = f"({', '.join(tuple_parts)})"
        ast_block.add_statement(
            ast.parse(f"np.add.at({node.name}, {index_expr}, {value.name})").body[0]
        )
    else:
        ast_block.add_subscript_assignment(
            lhs_name=node.name, rhs_name=value.name, indices=index_strs, lhs_is_indexed=True
        )
