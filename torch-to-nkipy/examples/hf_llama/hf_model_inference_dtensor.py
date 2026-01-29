import argparse

from hf_distributed_utils import initialize_distributed, load_parallelized_model, master_print
from hf_utils import compile_and_wrap_model, setup_neuron_environment
from transformers import AutoTokenizer

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
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
]


def main():
    """Main entry point for distributed inference."""
    parser = argparse.ArgumentParser(
        description="Run distributed inference with LLM on AWS Neuron"
    )
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

    args = parser.parse_args()

    # Initialize distributed environment
    rank, world_size, mesh = initialize_distributed()

    # Setup neuron environment with rank/world size
    setup_neuron_environment(rank=rank, world_size=world_size)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load parallelized model
    model = load_parallelized_model(args.model, args.dtype, rank, world_size, mesh)

    # Prepare input and generation config
    input_ids = tokenizer(args.prompt, return_tensors="pt")
    generation_kwargs = {"do_sample": False}
    generation_kwargs["max_length"] = args.max_length

    # Compile and wrap the model
    model_compiled = compile_and_wrap_model(model)

    master_print("Warming up the model...", flush=True)
    outputs_compiled = model_compiled.generate(**input_ids, **generation_kwargs)
    master_print("Model warmed up!", flush=True)

    master_print("Running generation...", flush=True)
    model_compiled._cache = None
    outputs_compiled = model_compiled.generate(**input_ids, **generation_kwargs)
    result = tokenizer.batch_decode(outputs_compiled, skip_special_tokens=True)
    master_print(f"Output (device):\n{result[0]}\n", flush=True)

if __name__ == "__main__":
    main()
