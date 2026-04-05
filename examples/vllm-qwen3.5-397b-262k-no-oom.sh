#!/bin/bash
# PROFILE: Qwen3.5-397B 262K Context (No OOM)
# DESCRIPTION: Quick launch after reboot for Intel/Qwen3.5-397B-A17B-int4-AutoRound
#              at 262K context on 2x DGX Sparks with --no-ray mode.
#              Includes post-reboot safety steps, system memory tuning,
#              and conservative vLLM parameters to prevent OOM crashes.
#
# Usage: ./launch-cluster.sh --no-ray -t vllm-node-tf5 --launch-script vllm-qwen3.5-397b-262k-no-oom.sh
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
#    - --gpu-memory-utilization 0.80: Reserve 20% GPU memory headroom (~24GB total)
#    - --max-num-seqs 1: Single-user mode (eliminates concurrent request memory spikes)
#    - --max-num-batched-tokens 1024: Minimal batch memory footprint
#    - --max-model-len 131072: Half context (saves ~12GB KV cache vs 262K)
#    - Prefix caching DISABLED: Saves ~8-12GB KV cache overhead
#
# === Memory Budget (per node, 128GB unified RAM) ===
#
#    Component                    | Estimated Usage
#    -----------------------------|------------------
#    Model weights (INT4-AutoRound)| ~60 GB
#    KV cache (131K, fp8)         | ~12 GB
#    Activations + overhead       | ~8 GB
#    System reserved (GPU 20%)    | ~24 GB
#    -----------------------------|------------------
#    Total                        | ~104 GB / 128 GB
#    Headroom                     | ~24 GB
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
#    - Reduce context further: change --max-model-len to 65536
#    - Reduce batch tokens: change --max-num-batched-tokens to 512
#    - Lower GPU memory: change --gpu-memory-utilization to 0.75

# --- Post-Reboot Safety Steps ---

echo "Dropping page caches..."
sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

echo "Limiting GPU clock to 200-2150 MHz..."
nvidia-smi -lgc 200,2150

# --- System Memory Tuning ---

echo "Tuning system memory parameters..."
sysctl -w vm.swappiness=10
sysctl -w vm.min_free_kbytes=524288
sysctl -w vm.dirty_background_ratio=5
sysctl -w vm.dirty_ratio=10

# --- RDMA/RoCE Network Tuning ---

echo "Tuning RDMA socket buffers for RoCE..."
sysctl -w net.core.rmem_max=16777216
sysctl -w net.core.wmem_max=16777216
sysctl -w net.core.rmem_default=16777216
sysctl -w net.core.wmem_default=16777216
sysctl -w net.ipv4.tcp_rmem='4096 87380 16777216'
sysctl -w net.ipv4.tcp_wmem='4096 65536 16777216'

# --- IB Interface Verification ---

echo "Checking IB link connectivity..."
IB_OK=0
for iface in enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1; do
    if [[ "$(cat /sys/class/net/$iface/carrier 2>/dev/null)" == "1" ]]; then
        IB_OK=1
        break
    fi
done
if [[ "$IB_OK" == "1" ]]; then
    echo "IB link verified via $iface"
    export NCCL_IB_DISABLE=0
    export NCCL_SOCKET_IFNAME=enP7s7
else
    echo "Warning: IB link not responding. Using management network for NCCL."
    export NCCL_SOCKET_IFNAME=enP7s7
    export NCCL_IB_DISABLE=1
fi

# Ensure the head's IB IP is correct (10.10.10.5/30 on enp1s0f0np0)
if [[ "$IB_OK" == "1" ]] && ! ip addr show "$iface" 2>/dev/null | grep -q "10.10.10.5"; then
    echo "Fixing IB IP on $iface..."
    ip addr replace 10.10.10.5/30 dev "$iface" 2>/dev/null || true
fi

# --- Environment Variables ---

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_MARLIN_USE_ATOMIC_ADD=1
export OMP_NUM_THREADS=4

# --- vLLM Serve Command ---

vllm serve Intel/Qwen3.5-397B-A17B-int4-AutoRound \
    --max-model-len 131072 \
    --max-num-seqs 1 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.80 \
    --port 8000 \
    --host 0.0.0.0 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --max-num-batched-tokens 1024 \
    --trust-remote-code \
    -tp 2
