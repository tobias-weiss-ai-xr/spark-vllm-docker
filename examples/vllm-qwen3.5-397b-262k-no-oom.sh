#!/bin/bash
# PROFILE: Qwen3.5-397B 262K Context (No OOM)
# DESCRIPTION: Quick launch after reboot for Intel/Qwen3.5-397B-A17B-int4-AutoRound
#              at 262K context on 2x DGX Sparks with --no-ray mode.
#              Includes post-reboot safety steps to prevent OOM and sudden shutdowns.
#
# Usage: ./launch-cluster.sh --no-ray --launch-script vllm-qwen3.5-397b-262k-no-oom.sh
# Required mods: --apply-mod mods/fix-qwen3.5-autoround --apply-mod mods/fix-qwen3.5-chat-template --apply-mod mods/gpu-mem-util-gb

# --- Post-Reboot Safety Steps ---

# 1. Drop page caches to prevent model getting stuck loading weights
echo "Dropping page caches..."
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

# 2. Limit GPU clock frequency to prevent sudden shutdowns during heavy inference
#    Default max is 2411 MHz (can boost to 3000 MHz). Capping at 2150 MHz for stability.
echo "Limiting GPU clock to 200-2150 MHz..."
sudo nvidia-smi -lgc 200,2150

# --- Environment Variables ---

# Enable expandable segments for better memory fragmentation handling
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use atomic add for Marlin kernels (required for INT4-AutoRound)
export VLLM_MARLIN_USE_ATOMIC_ADD=1

# Limit OpenMP threads to reduce memory pressure
export OMP_NUM_THREADS=4

# --- vLLM Serve Command ---

vllm serve Intel/Qwen3.5-397B-A17B-int4-AutoRound \
    --max-model-len 262144 \
    --max-num-seqs 2 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization-gb 110 \
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
