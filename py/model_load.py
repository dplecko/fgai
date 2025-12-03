MODEL_PATHS = {
    ### instruct versions
    "llama3_8b_instruct": (
        "/local/eb/dp3144/llama3_8b_instruct",
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
