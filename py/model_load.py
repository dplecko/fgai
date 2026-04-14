import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer


MODEL_PATHS = {
    # smaller models
    "llama3_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "ministral3_8b": "mistralai/Ministral-3-8B-Instruct-2512",
    "gemma3_4b": "google/gemma-3-4b-it",
    "qwen35_9b": "Qwen/Qwen3.5-9B",
    "deepseek_7b": "deepseek-ai/deepseek-llm-7b-chat",
    "phi4": "microsoft/phi-4",
    # larger models
    "qwen35_27b": "Qwen/Qwen3.5-27B",
    "gemma3_27b": "google/gemma-3-27b-it",
    "deepseek_r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "llama3_70b": "meta-llama/Llama-3.3-70B-Instruct",
}

VLLM_OVERRIDES = {
    "llama3_70b": {"max_model_len": 8192},
}
_PATH_TO_NAME = {v: k for k, v in MODEL_PATHS.items()}


def get_vllm_model(model_path, dtype="bfloat16", tensor_parallel_size=None):
    from vllm import LLM

    if tensor_parallel_size is None:
        tensor_parallel_size = torch.cuda.device_count() or 1
    kwargs = dict(
        model=model_path,
        dtype=dtype,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        max_logprobs=26,
    )
    name = _PATH_TO_NAME.get(model_path)
    if name:
        kwargs.update(VLLM_OVERRIDES.get(name, {}))
    return LLM(**kwargs)


# model loading
def get_model(model_path, dtype="auto"):
    kwargs = dict(
        torch_dtype=dtype,  # "auto" reads from config (fp8, bf16, etc.)
        device_map="auto",
        attn_implementation="sdpa",
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except ValueError:
        model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer, model.device


def model_batchsize(model_name):
    if model_name in [
        "llama3_8b",
        "gemma3_4b",
        "ministral3_8b",
        "qwen35_9b",
        "deepseek_7b",
        "phi4",
    ]:
        return 128
    elif model_name in [
        "qwen35_27b",
        "gemma3_27b",
        "deepseek_r1",
    ]:
        return 16
    elif model_name in ["llama3_70b"]:
        return 4
    else:
        raise ValueError(f"Unknown model name: {model_name}")
