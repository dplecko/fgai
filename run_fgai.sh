#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs

MODELS=(deepseek_r1 llama3_70b)
# MODELS=(llama3_8b ministral3_8b gemma3_4b qwen35_9b deepseek_7b phi4 qwen35_27b gemma3_27b deepseek_r1 llama3_70b)

for model in "${MODELS[@]}"; do
    echo "=== $model ==="
    python -m py.elicit --model "$model" --engine vllm --batch 1 \
        > "logs/elicit_${model}.log" 2>&1
done

echo "Done."
