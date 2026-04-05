# AGENTS.md - Project Context for AI Agents

## Quick Launch: Qwen3.5-397B on 2x DGX Spark

### Exact Launch Command

```bash
./launch-cluster.sh --no-ray \
  --launch-script vllm-qwen3.5-397b-262k-no-oom.sh
```

No mods required — script uses `--gpu-memory-utilization 0.85` (native flag).

### Prerequisites
- Container image `vllm-node-tf5` must exist on both nodes
- Model `Intel/Qwen3.5-397B-A17B-int4-AutoRound` downloaded to `~/.cache/huggingface`
- Passwordless SSH between nodes (192.168.0.27 ↔ 192.168.0.176)
- `.env` file present with cluster config

### Post-Reboot Safety (handled by launch script)
1. Drop page caches: `sync; echo 3 > /proc/sys/vm/drop_caches`
2. Limit GPU clock: `sudo nvidia-smi -lgc 200,2150`

### System Memory Tuning (handled by launch script)
- `vm.swappiness=10` — defer swap until RAM nearly exhausted
- `vm.min_free_kbytes=524288` — 512MB kernel memory floor
- `vm.dirty_background_ratio=5` — begin writeback at 5% dirty pages
- `vm.dirty_ratio=10` — force writeback at 10% dirty pages

### vLLM Parameters (conservative, OOM-safe)
| Parameter | Value | Reason |
|-----------|-------|--------|
| `--gpu-memory-utilization` | 0.85 | 15% headroom (~18GB) |
| `--max-num-seqs` | 1 | Single-user, no concurrent spikes |
| `--max-num-batched-tokens` | 2048 | Limits batch memory footprint |
| `--swap-space` | 16 | 16GB CPU swap buffer |
| `--kv-cache-dtype` | fp8 | Half precision KV cache |
| `--max-model-len` | 262144 | Full context length |

### Memory Budget (per node, 128GB unified RAM)

| Component | Estimated Usage |
|-----------|----------------|
| Model weights (INT4-AutoRound) | ~60 GB |
| KV cache (262K, fp8) | ~25 GB |
| Activations + overhead | ~10 GB |
| System reserved (GPU 15%) | ~18 GB |
| **Total** | **~113 GB / 128 GB** |
| **Headroom** | **~15 GB** |

### If Still OOM
1. Disable prefix caching: add `--no-prefix-caching` to vllm serve command
2. Reduce context: change `--max-model-len` to 131072
3. Reduce batch tokens: change `--max-num-batched-tokens` to 1024

### If Mods Fail (patch hunk rejected)
Rebuild container: `./build-and-copy.sh --tf5 --rebuild-vllm`

### Cluster Config (.env)
```
CLUSTER_NODES=192.168.0.27,192.168.0.176
COPY_HOSTS=192.168.0.176
ETH_IF=enP7s7
IB_IF=rocep1s0f0
LOCAL_IP=192.168.0.27
```

### Stop Cluster
```bash
./launch-cluster.sh stop
```

### Check Status
```bash
./launch-cluster.sh status
```
