#!/usr/bin/env bash
#SBATCH --account=a0181
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --job-name=gen_only
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --environment=fgai
#SBATCH --output=logs/sbatch_%A_%a.out
#SBATCH --error=logs/sbatch_%A_%a.err
#SBATCH --no-requeue
#SBATCH --array=0-0   # = len(GEN_MODELS) - 1; update if the list below changes

cd ~/fgai
set -euo pipefail
ulimit -c 0

# --- generation-only pass ---
# Pre-populates the attempt 1-4 story caches (data/cache/*_gen.parquet) for the
# 10 paper models, ahead of a separate --ann_only annotation pass. One array
# task = one generation model; no annotator model is loaded here at all.
# GEN_MODELS=(
#     llama3_8b ministral3_8b gemma3_4b qwen35_9b deepseek_7b phi4
#     qwen35_27b gemma3_27b deepseek_r1 llama3_70b
# )
GEN_MODELS=(ministral3_8b)

model=${GEN_MODELS[$SLURM_ARRAY_TASK_ID]}

# phi4 only uses 2 GPUs; every other model gets all 4
if [[ "$model" == "phi4" ]]; then
    export CUDA_VISIBLE_DEVICES=0,1
else
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

echo "=== gen_only model=$model ==="
python3 -m py.elicit --model "$model" --engine vllm --batch 1 --gen_only \
    > "logs/genonly_${model}.log" 2>&1

echo "Done."
