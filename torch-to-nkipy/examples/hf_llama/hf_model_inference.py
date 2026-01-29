"""HF model inference script for Neuron devices."""
import argparse

from hf_utils import (
    compile_and_wrap_model,
    load_model_and_tokenizer,
    setup_neuron_environment,
)

import warnings
from transformers import logging

warnings.filterwarnings("ignore", module="transformers")
logging.set_verbosity_error()

# Supported models
SUPPORTED_MODELS = [
    # Llama models
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
]


def main():
    """Main entry point for the inference script."""
    parser = argparse.ArgumentParser(description="Run inference with LLM on AWS Neuron")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        choices=SUPPORTED_MODELS,
        help=f"Name of the model to use. Valid options: {', '.join(SUPPORTED_MODELS)}",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="Model dtype (float32, bfloat16, or float16)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is Paris.",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-length", type=int, default=128, help="Maximum length for generation"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./nkipy_cache",
        help="Directory for nkipy cache",
    )

    args = parser.parse_args()

    # Setup the environment
    setup_neuron_environment(cache_dir=args.cache_dir)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=args.dtype)

    # Prepare input
    input_ids = tokenizer(args.prompt, return_tensors="pt")

    # Prepare generation config
    generation_kwargs = {"do_sample": False}
    generation_kwargs["max_length"] = args.max_length

    # Generate with the original model
    outputs = model.generate(**input_ids, **generation_kwargs)
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(f"Output (host):\n{result[0]}\n", flush=True)

    # Wrap and compile the model
    model_compiled = compile_and_wrap_model(model)

    print("Warming up the model...", flush=True)
    # Generate with compiled model
    outputs_compiled = model_compiled.generate(**input_ids, **generation_kwargs)
    print("Model warmed up!", flush=True)

    print("Running generation...", flush=True)
    model_compiled._cache = None
    outputs_compiled = model_compiled.generate(**input_ids, **generation_kwargs)
    result = tokenizer.batch_decode(outputs_compiled, skip_special_tokens=True)
    print(f"Output (device):\n{result[0]}\n", flush=True)


if __name__ == "__main__":
    main()
