# Torch-to-NKIPy

Torch-to-NKIPy enables running PyTorch models on AWS Trainium accelerators through seamless integration with `torch.compile` and NKIPy. It provides a PyTorch Dynamo backend that automatically converts PyTorch operations to NKIPy kernels for execution on Trainium hardware.

## Features

- **Seamless Integration**: Drop-in replacement for PyTorch's default backend via `torch.compile(backend="nkipy")`
- **Fine-grained Control**: Mark subgraphs for selective compilation and control CPU/Trainium execution boundaries
- **Custom NKI Kernels**: Inject optimized custom NKI (Neuron Kernel Interface) kernels for enhanced performance
- **Comprehensive Op Support**: Extensive ATen operation registry with 50+ operations including:
  - Tensor operations (add, mul, matmul, etc.)
  - Activation functions (relu, gelu, sigmoid, etc.)
  - Normalization (layer_norm, softmax, etc.)
  - Indexing and slicing operations
  - Collective operations for distributed training
- **Automatic Tensor Management**: Handles runtime tensor placement and data movement between CPU and Trainium devices
- **Custom Device Registration**: C++ extension for PyTorch device integration with Neuron Runtime (NRT)

## Architecture

The package consists of several key components:

1. **Backend** (`backend/`): PyTorch Dynamo backend registration and graph compilation
2. **NKIPy Builder** (`nkipy_builder/`): Converts FX graphs to NKIPy AST and generates executable kernels
3. **ATen Op Registry** (`nkipy_builder/aten_op_registry/`): Maps PyTorch ATen operations to NKIPy implementations
4. **Device Module** (`device/`): Custom device registration and distributed backend support
5. **Runtime** (`runtime/`): Execution runtime for compiled kernels
6. **C++ Extension** (`csrc/`): Native PyTorch device registration and tensor allocation

## Quick Start

### Basic Usage

```python
import torch
import torch.nn as nn
from torch_to_nkipy import init_nkipy_backend

# Initialize the NKIPy backend
init_nkipy_backend()

# Define and compile your model
@torch.compile(backend="nkipy", fullgraph=True, dynamic=False)
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8, bias=False)
        self.fc2 = nn.Linear(8, 4, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.fc2(self.act_fn(self.fc1(x)))

# Move model and input to nkipy device
model = MLP().to("nkipy")
input_tensor = torch.randn(2, 4).to("nkipy")

# Run inference
with torch.no_grad():
    output = model(input_tensor)
print(output.cpu())
```

### Hybrid Execution (CPU + Trainium)

```python
# Parts of the model can run on CPU, others on Trainium
model = MLP().to("nkipy")

def forward(x):
    x = trainium_layer(x)      # Runs on Trainium
    x = x.cpu()                 # Transfer to CPU
    x = cpu_operation(x)        # Runs on CPU
    x = x.to("nkipy")          # Transfer back to Trainium
    x = trainium_layer2(x)      # Runs on Trainium
    return x
```

### Custom NKI Kernel Integration

```python
import torch
from torch_to_nkipy import init_nkipy_backend, NKIOpRegistry

init_nkipy_backend()

# Register custom NKI kernel implementation
@NKIOpRegistry.register("mylib::custom_add")
def nki_custom_add_kernel(a_input, b_input):
    # NKI kernel implementation here
    # ...
    return c_output

# Register as PyTorch custom operator
@torch.library.custom_op("mylib::custom_add", mutates_args=())
def custom_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return nki_custom_add_kernel(a, b)

# Use in compiled code
@torch.compile(backend="nkipy", fullgraph=True, dynamic=False)
def my_function(a, b):
    return custom_add(a, b)
```

## Configuration

The backend can be configured through `init_nkipy_backend()`:

- `nkipy_cache`: Cache directory for compiled kernels (default: `"./nkipy_cache"`)
- `log_level`: Logging verbosity (default: `logging.INFO`)
- `rank`: Process rank for distributed execution (default: `0`)
- `world_size`: Total number of processes (default: `1`)
- `additional_compiler_args`: Custom compiler flags (default: `""`)

## Supported Operations

The package includes implementations for 50+ PyTorch operations including:

**Arithmetic**: add, sub, mul, div, pow, abs, reciprocal, sqrt, rsqrt

**Linear Algebra**: matmul, addmm, embedding

**Activations**: relu, gelu, sigmoid, hardtanh, leaky_relu, log_softmax, softmax

**Normalization**: native_layer_norm, linalg_vector_norm

**Reductions**: sum, mean, max, min, any, argmax, argmin

**Indexing/Slicing**: index, select, slice, gather, index_select, squeeze

**Shape Operations**: expand, repeat, cat, split_with_sizes, view, permute

**Tensor Creation**: full, arange, scalar_tensor, empty_permuted

**Logical**: where, clamp, ne, bitwise_not

**Collectives**: all_reduce, all_gather (for distributed training)

See `nkipy_builder/aten_op_registry/` for the complete list.

## Development

### Project Structure

```
torch-to-nkipy/
├── src/torch_to_nkipy/           # Python package
│   ├── backend/                   # Dynamo backend implementation
│   ├── device/                    # Device module and distributed support
│   ├── nkipy_builder/            # FX graph to NKIPy conversion
│   │   └── aten_op_registry/     # ATen operation implementations
│   ├── runtime/                   # Execution runtime
│   └── utils/                     # Utility functions
├── csrc/                          # C++ extension source
│   ├── nkipy_device.cpp/h        # Device registration
│   ├── nkipy_tensor.cpp/h        # Tensor implementation
│   ├── nkipy_tensor_allocator.cpp/h  # Memory allocation
│   ├── python_bindings.cpp        # PyBind11 bindings
│   └── torch_register.cpp/h       # PyTorch integration
├── setup.py                       # Build configuration
├── pyproject.toml                 # Package metadata
└── MANIFEST.in                    # Source distribution manifest
```
