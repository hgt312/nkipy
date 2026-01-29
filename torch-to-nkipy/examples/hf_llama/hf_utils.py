from typing import Any, Callable, Optional, Tuple, Union
import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StaticCache,
)

from torch_to_nkipy import init_nkipy_backend


def move_to_device(arg: Any, device: Optional[Union[str, torch.device]] = None) -> Any:
    if device is None:
        device = "cpu"

    if isinstance(arg, torch.Tensor):
        return arg.to(device)
    elif isinstance(arg, StaticCache):
        arg.key_cache = move_to_device(arg.key_cache, device)
        arg.value_cache = move_to_device(arg.value_cache, device)
        return arg
    elif isinstance(arg, (list, tuple)):
        return type(arg)(move_to_device(item, device) for item in arg)
    elif isinstance(arg, dict):
        return {k: move_to_device(v, device) for k, v in arg.items()}
    else:
        return arg


def forward_wrap(compiled_forward: Callable, *args: Any, **kwargs: Any) -> Any:
    # Set target device based on model type
    target_device = "nkipy"

    # Move inputs to target device
    args = [move_to_device(arg, target_device) for arg in args]
    kwargs = {k: move_to_device(v, target_device) for k, v in kwargs.items()}

    # Run the compiled forward pass
    out = compiled_forward(*args, **kwargs)

    # Move outputs back to CPU
    if hasattr(out, "logits"):
        out.logits = out.logits.cpu()

    return out


def setup_neuron_environment(
    rank: int = 0, world_size: int = 1, cache_dir: str = None
) -> None:
    # Initialize the Neuron backend
    cache_path = "./nkipy_cache" if cache_dir is None else cache_dir
    log_level = logging.INFO if rank == 0 else logging.ERROR
    init_nkipy_backend(nkipy_cache=cache_path, rank=rank, world_size=world_size, log_level=log_level)


def compile_and_wrap_model(
    model: Any,
) -> Any:
    # Wrap the forward method
    compiled_forward = torch.compile(
        model.forward, backend="nkipy", dynamic=False, fullgraph=True
    )

    # Create a wrapper function that uses the compiled forward
    def model_forward_wrap(*args: Any, **kwargs: Any) -> Any:
        return forward_wrap(compiled_forward, *args, **kwargs)

    # Replace the model's forward method
    model.forward = model_forward_wrap

    # Move the model to the target device
    model.to("nkipy")

    return model

def load_model_and_tokenizer(
    model_name: str,
    dtype: str = "float32",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.generation_config.cache_implementation = "static"
    model.config._attn_implementation = "eager"

    return model, tokenizer
