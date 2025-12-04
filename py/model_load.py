import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATHS = {
    ### instruct versions
    "llama3_8b_instruct": (
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        True,
    ),  # LLaMA 3.1 8B-Instruct
    "llama3_70b_instruct": (
        "/local/eb/dp3144/llama3_70b_instruct",
        True,
    ),  # LLaMA 3.3 70B-Instruct
    "mistral_7b_instruct": (
        "/local/eb/dp3144/mistral_7b_instruct",
        True,
    ),  # Instruct Mistral
    "phi4": ("/local/eb/dp3144/phi4", True),  # Microsoft Phi-4
    "gemma3_27b_instruct": (
        "/local/eb/dp3144/gemma3_27b_instruct",
        True,
    ),  # Gemma 27B-Instruct
    "deepseek_7b_chat": (
        "/local/eb/dp3144/deepseek_7b_chat",
        True,
    ),  # Instruct DeepSeek
    ### non-instruct versions
    "llama3_8b": ("/local/eb/dp3144/llama3_8b", False),  # Regular LLaMA 3 8B
    "llama3_70b": ("/local/eb/dp3144/llama3_70b", False),  # Regular LLaMA 3 8B
    "mistral_7b": ("/local/eb/dp3144/mistral_7b", False),  # Regular Mistral
    "gemma3_27b": ("/local/eb/dp3144/gemma3_27b", False),  # Gemma 27B-Instruct
    "deepseek_7b": ("/local/eb/dp3144/deepseek_7b", False),  # Regular DeepSeek
    "gpt2": ("/local/eb/dp3144/gpt2", False),  # Regular GPT-2
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
