# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training example using the spiky backend with explicit bucket sizes.

Trains a small MLP for several steps with varying batch sizes on Neuron
hardware and compares with eager CPU results. Uses explicit bucket sizes
via torch.compile options instead of relying on auto-inferred buckets.

Usage:
    cd nkipy/
    uv run python experimental/spiky/spiky/examples/example_training.py
"""

import copy

import torch
import torch.nn as nn

try:
    # Prefer spiky backend init (registers torch.compile backend "nkipy").
    from spiky.torch.backend import init_nkipy_backend
except Exception:
    from spiky.torch import init_nkipy_backend

torch.manual_seed(0)

init_nkipy_backend(nkipy_cache="./nkipy_cache")

# Define model
model = nn.Sequential(
    nn.Linear(32, 64, bias=False),
    nn.ReLU(),
    nn.Linear(64, 16, bias=False),
)

# Eager reference (CPU)
ref_model = copy.deepcopy(model)
ref_opt = torch.optim.SGD(ref_model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Compiled (NKIPy) with explicit bucket sizes for the batch dimension.
# Buckets [4, 8, 16] cover all batch sizes used below. Inputs are padded
# to the nearest bucket size before execution on device.
comp_model = copy.deepcopy(model).to("nkipy")
comp_opt = torch.optim.SGD(comp_model.parameters(), lr=0.01, foreach=False)


@torch.compile(
    backend="nkipy", fullgraph=True, dynamic=True,
    options={"buckets": [4, 8, 16]},
)
def forward_with_loss(m, x, target):
    out = m(x)
    loss = loss_fn(out, target)
    return out, loss


compiled_opt_step = torch.compile(comp_opt.step, backend="nkipy")

# Train with varying batch sizes that trigger padding to the nearest
# explicit bucket (e.g. 5→8, 7→8, 10→16, 13→16).
batch_sizes = [5, 7, 5, 13, 10]
for step, batch_size in enumerate(batch_sizes):
    torch.manual_seed(step)
    x = torch.randn(batch_size, 32)
    target = torch.randn(batch_size, 16)

    # Mark batch dimension as dynamic
    torch._dynamo.maybe_mark_dynamic(x, 0)
    torch._dynamo.maybe_mark_dynamic(target, 0)

    # Eager
    ref_opt.zero_grad()
    ref_loss = loss_fn(ref_model(x), target)
    ref_loss.backward()
    ref_opt.step()

    # Compiled
    comp_opt.zero_grad()
    _, comp_loss = forward_with_loss(comp_model, x.to("nkipy"), target.to("nkipy"))
    comp_loss.backward()
    compiled_opt_step()

    print(
        f"Step {step} (batch={batch_size}): "
        f"eager_loss={ref_loss.item():.4f}  "
        f"compiled_loss={comp_loss.cpu().item():.4f}"
    )

print("Training complete!")
