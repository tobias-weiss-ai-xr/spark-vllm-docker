#!/bin/bash
# PROFILE: Qwen3.5-397B 262K Context (No OOM)
# DESCRIPTION: Quick launch after reboot for Intel/Qwen3.5-397B-A17B-int4-AutoRound
#              at 262K context on 2x DGX Sparks with --no-ray mode.
#              Includes post-reboot safety steps, system memory tuning,
#              and conservative vLLM parameters to prevent OOM crashes.
#
# Usage: ./launch-cluster.sh --no-ray --launch-script vllm-qwen3.5-397b-262k-no-oom.sh
#
# === OOM Prevention Strategy ===
#
# This script applies three layers of protection against out-of-memory crashes:
#
# 1. POST-REBOOT CLEANUP
#    - Drops Linux page caches (frees memory held by filesystem cache)
#    - Caps GPU clock at 2150 MHz (prevents firmware-triggered shutdowns under load)
#
# 2. SYSTEM MEMORY TUNING (sysctl)
#    - vm.swappiness=10: Defer swap usage until RAM is nearly exhausted
#    - vm.min_free_kbytes=524288 (512MB): Reserve kernel memory floor
#    - vm.dirty_background_ratio=5: Begin background writeback at 5% dirty pages
#    - vm.dirty_ratio=10: Force synchronous writeback at 10% dirty pages
#
# 3. CONSERVATIVE vLLM PARAMETERS
#    - --gpu-memory-utilization 0.85: Reserve 15% GPU memory headroom (~18GB total)
#    - --max-num-seqs 1: Single-user mode (eliminates concurrent request memory spikes)
#    - --max-num-batched-tokens 2048: Limits batch memory footprint
#    - --swap-space 16: 16GB CPU swap buffer for vLLM KV cache overflow
#    - --enable-prefix-caching: Reuses KV cache for repeated prompts (saves memory on
#      multi-turn conversations; disable with --no-prefix-caching if still OOM)
#
# === Memory Budget (per node, 128GB unified RAM) ===
#
#    Component                    | Estimated Usage
#    -----------------------------|------------------
#    Model weights (INT4-AutoRound)| ~60 GB
#    KV cache (262K, fp8)         | ~25 GB
#    Activations + overhead       | ~10 GB
#    System reserved (GPU 15%)    | ~18 GB
#    -----------------------------|------------------
#    Total                        | ~113 GB / 128 GB
#    Headroom                     | ~15 GB
#
# === Known Issues ===
#
# 1. Sudden shutdowns during heavy inference
#    Fix: GPU clock limiting (applied automatically by this script)
#    Command: sudo nvidia-smi -lgc 200,2150
#
# 2. Model gets stuck loading weights
#    Fix: Drop page caches before launch (applied automatically by this script)
#    Command: sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
#
# 3. Still OOM after this script?
#    - Disable prefix caching: add --no-prefix-caching to vllm serve command
#    - Reduce context: change --max-model-len to 131072
#    - Reduce batch tokens: change --max-num-batched-tokens to 1024

# --- Post-Reboot Safety Steps ---

echo "Dropping page caches..."
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

echo "Limiting GPU clock to 200-2150 MHz..."
sudo nvidia-smi -lgc 200,2150

# --- System Memory Tuning ---

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
    --max-num-seqs 1 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.85 \
    --swap-space 16 \
    --port 8000 \
    --host 0.0.0.0 \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --max-num-batched-tokens 2048 \
    --trust-remote-code \
    --chat-template unsloth.jinja \
    -tp 2 \
    --distributed-executor-backend ray
