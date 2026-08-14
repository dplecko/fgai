#!/usr/bin/env bash
#SBATCH --account=a0181
#SBATCH --partition=normal
#SBATCH --time=01:30:00
#SBATCH --job-name=goldtest_405b_mp
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/sbatch_%j.out
#SBATCH --error=logs/sbatch_%j.err
#SBATCH --no-requeue

# --- goldtest_structured_http on Llama-3.1-405B via `vllm serve` with the
# native multiprocessing executor (--distributed-executor-backend mp), no
# Ray at all. 4 nodes x 4 GPUs = 16 GPUs as tensor-parallel-size=4 (fits
# within one node's 4 GPUs) x pipeline-parallel-size=4 (one stage per node),
# matching vLLM's documented multi-node mp pattern (TP within a node's GPU
# count, PP across nodes) -- unlike Ray, PP-across-nodes is plain
# torch.distributed point-to-point communication, not Ray's GCS/placement
# groups, which is where every prior Ray attempt this session got stuck.
#
# Genuinely untested end-to-end -- first real run is the test, same as every
# other script this session.
cd ~/fgai
set -euo pipefail
ulimit -c 0

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes[0]}
head_node_ip=$(srun --environment=fgai --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')
echo "Head node: $head_node ($head_node_ip)"

echo "Installing openai client (shared \$HOME, once is enough)..."
srun --environment=fgai --nodes=1 --ntasks=1 -w "$head_node" pip install --user --quiet openai

MODEL_PATH="meta-llama/Llama-3.1-405B-Instruct"
SERVED_NAME="llama3_405b"
PORT=8000

# vLLM's own compile cache (~/.cache/vllm) and Triton's separate JIT kernel
# cache both default under $HOME, which is Lustre-backed here -- a known
# upstream vLLM issue (github.com/vllm-project/vllm/issues/24601, #6180):
# filelocks are unreliable over NFS/Lustre, so concurrent processes racing to
# write/read the same cache entry hit ESTALE.
#
# First attempt redirected both to /dev/shm -- node-local, off Lustre, but
# tmpfs is very commonly mounted noexec, and Triton compiles a native .so it
# then mmaps as executable code (cuda_utils...so); that combination failed
# with "failed to map segment from shared object". /tmp is still node-local
# (not Lustre) and, unlike /dev/shm, almost always permits exec.
CACHE_ENV='export VLLM_CACHE_ROOT=/tmp/vllm_cache TRITON_CACHE_DIR=/tmp/triton_cache;'

# --enforce-eager was dropped for one run (2026-08-13) to test whether it was
# still needed now that the cache crash above is fixed by the redirect. It
# isn't needed for correctness, but with it off the KV-cache sizing pass
# under-predicted peak activation memory once real concurrent requests hit
# shapes near the top of the compiled range (max_num_batched_tokens=8192) --
# two GPUs on one PP stage OOM'd a minute after startup ("CUDA out of memory
# ... this process has 94.75 GiB [of 95] memory in use", failing on a plain
# (tokens, 16384) activation buffer, unrelated to attention kernel choice).
# Back to eager for reliability; revisiting compiled mode is a separate
# follow-up (e.g. lower --gpu-memory-utilization to leave more headroom).

echo "Starting vllm serve head (rank 0) on $head_node..."
srun --environment=fgai --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "$CACHE_ENV cd ~/fgai && vllm serve '$MODEL_PATH' --served-model-name '$SERVED_NAME' \
        --tensor-parallel-size 4 --pipeline-parallel-size 4 \
        --distributed-executor-backend mp \
        --nnodes 4 --node-rank 0 --master-addr '$head_node_ip' \
        --port $PORT --trust-remote-code --dtype bfloat16 --max-logprobs 26 --enforce-eager" \
    > logs/vllm_serve_head.log 2>&1 &

rank=1
for worker in "${nodes[@]:1}"; do
    echo "Starting vllm serve worker (rank $rank) on $worker..."
    srun --environment=fgai --nodes=1 --ntasks=1 -w "$worker" \
        bash -c "$CACHE_ENV cd ~/fgai && vllm serve '$MODEL_PATH' --served-model-name '$SERVED_NAME' \
            --tensor-parallel-size 4 --pipeline-parallel-size 4 \
            --distributed-executor-backend mp \
            --nnodes 4 --node-rank $rank --master-addr '$head_node_ip' \
            --port $PORT --trust-remote-code --dtype bfloat16 --max-logprobs 26 --enforce-eager --headless" \
        > "logs/vllm_serve_worker${rank}.log" 2>&1 &
    rank=$((rank + 1))
done

# Shard loading reads the 756 GiB checkpoint cold off Lustre (vLLM's own log:
# "exceeds 90% of available RAM -- skipping auto-prefetch"), with highly
# variable per-shard latency under contention -- 10 min wasn't enough margin
# last run. 45 min still leaves most of the 90-min job for the actual query
# pass (300 rows, concurrency=32, normally a few minutes).
echo "Waiting for the server to become ready (up to 45 min)..."
srun --environment=fgai --overlap --nodes=1 --ntasks=1 -w "$head_node" bash -c "
deadline=\$((SECONDS + 2700))
while [ \$SECONDS -lt \$deadline ]; do
    if curl -sf http://$head_node_ip:$PORT/v1/models > /dev/null 2>&1; then
        echo 'Server is ready.'
        exit 0
    fi
    sleep 10
done
echo 'Timed out waiting for the server.' >&2
exit 1
"

echo "=== goldtest_structured_http ann=llama3_405b ==="
srun --environment=fgai --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "export PATH=\"\$HOME/.local/bin:\$PATH\"; cd ~/fgai && python3 -m py.goldtest_structured_http --few_shot \
        --server_url 'http://$head_node_ip:$PORT/v1' --ann_model '$SERVED_NAME' \
        --served_model_name '$SERVED_NAME'" \
    > "logs/goldtest_llama3_405b_structured_http_fewshot.log" 2>&1

echo "Done."
