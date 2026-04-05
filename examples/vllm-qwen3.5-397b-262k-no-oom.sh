#!/bin/bash
# PROFILE: Qwen3.5-397B 262K Context (No OOM)
# DESCRIPTION: Quick launch after reboot for Intel/Qwen3.5-397B-A17B-int4-AutoRound
#              at 262K context on 2x DGX Sparks with --no-ray mode.
#              Includes post-reboot safety steps to prevent OOM and sudden shutdowns.
#
# Usage: ./launch-cluster.sh --no-ray --launch-script vllm-qwen3.5-397b-262k-no-oom.sh

# --- Post-Reboot Safety Steps ---

echo "Dropping page caches..."
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

echo "Limiting GPU clock to 200-2150 MHz..."
sudo nvidia-smi -lgc 200,2150

# --- System Memory Tuning (OOM Prevention) ---

echo "Tuning system memory parameters..."
sudo sysctl -w vm.swappiness=10
sudo sysctl -w vm.min_free_kbytes=524288
sudo sysctl -w vm.dirty_background_ratio=5
sudo sysctl -w vm.dirty_ratio=10

# --- Environment Variables ---

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export OMP_NUM_THREADS=4

# --- vLLM Serve Command ---

vllm serve Intel/Qwen3.5-397B-A17B-int4-AutoRound \
    --max-model-len 262144 \
    --max-num-seqs 2 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.88 \
    --swap-space 16 \
    --port 8000 \
    --host 0.0.0.0 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --max-num-batched-tokens 4176 \
    --trust-remote-code \
    --chat-template unsloth.jinja \
    -tp 2 \
    --distributed-executor-backend ray
