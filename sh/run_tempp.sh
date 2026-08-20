#!/usr/bin/env bash
#SBATCH --account=a0181
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --job-name=tempp_sens
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --environment=fgai
#SBATCH --output=logs/sbatch_%A_%a.out
#SBATCH --error=logs/sbatch_%A_%a.err
#SBATCH --no-requeue
#SBATCH --array=0-3   # = len(GEN_MODELS) * len(TEMPS) - 1; update if the lists below change

cd ~/fgai
set -euo pipefail
ulimit -c 0

# --- temperature / top_p sensitivity grid ---
# small subset of gen models spanning the size range; annotator held fixed (single
# model, not swept) so the sweep isolates the generation-side effect only.
GEN_MODELS=(qwen35_27b)
ANN_MODEL=qwen25_72b

# T x p grid, (1.0, 1.0) dropped: it's the default, already generated/annotated, so
# re-running it here would just redo annotation on unchanged output.
TEMPS=(0.5 0.5 0.7 0.7)
TOPPS=(0.9 1.0 0.9 1.0)

export CUDA_VISIBLE_DEVICES=0,1,2,3

# one array task = one (gen_model, T, p) pair; GEN_MODELS is the fastest-varying
# (parallel) index, then the T/p combo (parallel arrays, indexed together).
n_gen=${#GEN_MODELS[@]}
gen_idx=$(( SLURM_ARRAY_TASK_ID % n_gen ))
tp_idx=$(( SLURM_ARRAY_TASK_ID / n_gen ))
model=${GEN_MODELS[$gen_idx]}
temp=${TEMPS[$tp_idx]}
topp=${TOPPS[$tp_idx]}

echo "=== gen=$model T=$temp p=$topp ==="
python3 -m py.elicit --model "$model" --ann_model "$ANN_MODEL" --engine vllm --batch 1 \
    --temperature "$temp" --top_p "$topp" \
    > "logs/elicit_${model}_t${temp}_p${topp}_${SLURM_JOB_ID}.log" 2>&1

echo "Done."
