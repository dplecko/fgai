import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATHS = {
    "llama3_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "ministral_3b": "mistralai/Ministral-3-8B-Instruct-2512",
    "gemma3_4b": "google/gemma-3-4b-it",
    "qwen3_9b": "Qwen/Qwen3.5-9B",
    "deepseek_7b": "deepseek-ai/deepseek-llm-7b-chat",
    "phi4": "microsoft/phi-4",
    # large models
    # "llama3_70b":    "meta-llama/Llama-3.3-70B-Instruct",
    # "gemma3_27b":    "google/gemma-3-27b-it",
    # "qwen3_35b":     "Qwen/Qwen3.5-35B-A3B",
    # "deepseek_r1":   "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
}


# model loading utilities
def get_device(prefer_gpu_idx: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{prefer_gpu_idx}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model(model_path, prefer_gpu_idx: int = 0):
    device = get_device(prefer_gpu_idx)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # change to "flash_attention_2" if your stack supports it
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding is typically faster for decoder-only models in batched generate
    tokenizer.padding_side = "left"

    return model, tokenizer, device
