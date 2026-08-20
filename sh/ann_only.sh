#!/usr/bin/env bash
#SBATCH --account=a0181
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --job-name=ann_all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --environment=fgai
#SBATCH --output=logs/sbatch_%A_%a.out
#SBATCH --error=logs/sbatch_%A_%a.err
#SBATCH --no-requeue
#SBATCH --array=0-3   # = len(GEN_MODELS) * len(ANN_MODELS) - 1; update if the lists below change

cd ~/fgai
set -euo pipefail
ulimit -c 0

# --- annotation-only pass ---
# Reads the attempt 1-4 story caches populated by gen_only.sh and annotates
# them; --ann_only never loads a generation model and errors out if a cache is
# missing, rather than silently generating on the fly. No --style/--temperature
# /--top_p flags here: they must match gen_only.sh's defaults exactly, since
# they determine which cache filenames get looked up.
# Add more entries to ANN_MODELS to run several annotators (cross-annotator
# agreement) over the same cached stories.
GEN_MODELS=(
    llama3_8b ministral3_8b ministral3_14b qwen35_9b mistral_24b phi4
    qwen35_27b gemma3_27b deepseek_r1 llama3_70b
)
ANN_MODELS=(llama3_70b commanda_111b) 
# one array task = one (gen_model, ann_model) pair; GEN_MODELS is the
# fastest-varying (parallel) index
n_gen=${#GEN_MODELS[@]}
gen_idx=$(( SLURM_ARRAY_TASK_ID % n_gen ))
ann_idx=$(( SLURM_ARRAY_TASK_ID / n_gen ))
model=${GEN_MODELS[$gen_idx]}
ann_model=${ANN_MODELS[$ann_idx]}

# GPU count depends on the annotator model being loaded (no generation model
# is ever loaded in --ann_only mode)
if [[ "$ann_model" == "phi4" ]]; then
    export CUDA_VISIBLE_DEVICES=0,1
else
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

# vLLM's compile cache defaults under $HOME (Lustre-backed here). With every
# array task potentially loading the *same* ann_model concurrently across
# different nodes, they race to read/write the same cache path and hit
# "OSError: [Errno 116] Stale file handle" -- same root cause diagnosed for
# the 405B multi-node run (goldtest_405b_mp.sh's CACHE_ENV). Redirected to
# /tmp here too: node-local, off Lustre, avoids the race entirely.
export VLLM_CACHE_ROOT=/tmp/vllm_cache
export TRITON_CACHE_DIR=/tmp/triton_cache

echo "=== ann_only gen=$model ann=$ann_model ==="
python3 -m py.elicit --model "$model" --ann_model "$ann_model" --engine vllm --batch 1 --ann_only --few_shot \
    > "logs/annonly_${model}_${ann_model}.log" 2>&1

echo "Done."
