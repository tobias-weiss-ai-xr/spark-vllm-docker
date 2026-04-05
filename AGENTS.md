# AGENTS.md - Project Context for AI Agents

## Quick Launch: Qwen3.5-397B on 2x DGX Spark

### Exact Launch Command

```bash
export VLLM_SPARK_EXTRA_DOCKER_ARGS="-v /tmp/vllm-compile-cache:/tmp/torchinductor_root"
nohup ./launch-cluster.sh --no-ray -t vllm-node-tf5 \
  --eth-if enP7s7 \
  --launch-script vllm-qwen3.5-397b-262k-no-oom.sh > /tmp/launch.log 2>&1 &
disown
```

The `VLLM_SPARK_EXTRA_DOCKER_ARGS` mounts a persistent triton compile cache so
subsequent launches skip the ~5 min cold compile. First launch will compile and
the API will be ready after ~10 min total (4.5 min model load + 5 min compile).

### Prerequisites
- Container image `vllm-node-tf5` must exist on both nodes
- Model `Intel/Qwen3.5-397B-A17B-int4-AutoRound` downloaded to `~/.cache/huggingface`
- Passwordless SSH between nodes (192.168.0.27 ↔ 192.168.0.176)
- Passwordless sudo on both nodes (for sysctl tuning)
- `.env` file present with cluster config

### Post-Reboot Checklist
1. Bring down phantom IB interface on both nodes: `sudo ip link set enP2p1s0f0np0 down`
2. Verify IB IP on head: `ip addr show enp1s0f0np0` should show `10.10.10.5/30`
3. Run the launch command above

### Post-Reboot Safety (handled by launch script)
1. Drop page caches: `sync; echo 3 > /proc/sys/vm/drop_caches`
2. Limit GPU clock: `nvidia-smi -lgc 200,2150`

### System Memory Tuning (handled by launch script)
- `vm.swappiness=10` — defer swap until RAM nearly exhausted
- `vm.min_free_kbytes=524288` — 512MB kernel memory floor
- `vm.dirty_background_ratio=5` — begin writeback at 5% dirty pages
- `vm.dirty_ratio=10` — force writeback at 10% dirty pages

### RDMA/RoCE Tuning (handled by launch script)
- `net.core.rmem_max` / `wmem_max` = 16MB
- `net.ipv4.tcp_rmem` / `tcp_wmem` max = 16MB

### vLLM Parameters
| Parameter | Value | Reason |
|-----------|-------|--------|
| `--gpu-memory-utilization` | 0.92 | Compiled CUDA graphs are resident in GPU memory |
| `--max-num-seqs` | 1 | Single-user, no concurrent spikes |
| `--max-num-batched-tokens` | 1024 | Limits batch memory footprint |
| `--kv-cache-dtype` | fp8 | Half precision KV cache |
| `--max-model-len` | 131072 | Half context (131K vs 262K) |
| `-tp` | 2 | Tensor parallel across 2 nodes |

### Memory Budget (per node, 128GB unified RAM)

| Component | Estimated Usage |
|-----------|----------------|
| Model weights (INT4-AutoRound) | ~60 GB |
| KV cache (131K, fp8) | ~15 GB |
| Compiled CUDA graphs | ~5 GB |
| Activations + overhead | ~8 GB |
| System reserved (GPU 8%) | ~10 GB |
| **Total** | **~98 GB / 128 GB** |
| **Headroom** | **~30 GB** |

### If Still OOM
1. Reduce context: change `--max-model-len` to 65536
2. Reduce batch tokens: change `--max-num-batched-tokens` to 512

### If Mods Fail (patch hunk rejected)
Rebuild container: `./build-and-copy.sh --tf5 --rebuild-vllm`

### Cluster Config (.env)
```
CLUSTER_NODES=192.168.0.27,192.168.0.176
COPY_HOSTS=192.168.0.176
ETH_IF=enP7s7
MASTER_PORT=29500
LOCAL_IP=192.168.0.27
```

### Network Topology

The two DGX Sparks are connected via two networks:

| Network | Interface | IPs | Purpose |
|---------|-----------|-----|---------|
| Management | `enP7s7` | 192.168.0.27 / 192.168.0.176 | SSH, NCCL bootstrap/GLOO coordination |
| InfiniBand (RoCEv2) | `enp1s0f0np0` | 10.10.10.5 / 10.10.10.6 | NCCL data plane (200Gbps HDR) |

#### IB Cable Requirements
- Direct cable between matching ports: head `enp1s0f0np0` ↔ worker `enp1s0f0np0`
- Both nodes use `/30` subnet (10.10.10.4/30): head .5, worker .6
- Only one IB cable should be connected per Spark. The second CX-7 NIC (`enP2p1s0f*`) shows phantom carrier=1 on some Sparks — it must be brought down or NCCL will discover it and fail:
  ```bash
  sudo ip link set enP2p1s0f0np0 down  # on both nodes, after every reboot
  ```

#### Post-Reboot IB Setup
The IB IP on the head is not persistent across reboots. The launch script auto-assigns it, but if you need to verify manually:
```bash
# On head (192.168.0.27):
sudo ip addr add 10.10.10.5/30 dev enp1s0f0np0
sudo ip link set enP2p1s0f0np0 down   # kill phantom interface

# On worker (192.168.0.176):
sudo ip link set enP2p1s0f0np0 down   # kill phantom interface
# Worker IP (10.10.10.6) is persistent
```

#### How NCCL Networking Works
- `ETH_IF` (enP7s7) — used for NCCL bootstrap, GLOO, and TCP fallback (`NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`)
- IB/RoCE — auto-detected by NCCL when `NCCL_IB_DISABLE=0` (the launch script sets this based on carrier detect)
- `IB_IF` in `.env` — **do not set**. `launch-cluster.sh` passes `NCCL_IB_HCA=$IB_IF` to containers. When empty, NCCL auto-discovers active RoCE devices. Setting it to a specific device name fails on nodes where that device doesn't exist
- `autodiscover.sh` requires `ibdev2netdev` to detect IB interfaces. If `ETH_IF` is set but `IB_IF` is not, IB detection is skipped (NCCL auto-detects instead)

#### Troubleshooting IB
1. `NCCL_IB : No device found` — IB link check failed (container has no `ping`). The launch script uses `/sys/class/net/$iface/carrier` instead
2. `ibv_modify_qp failed with 22 Invalid argument` — GID mismatch between nodes. Usually caused by multiple active ports confusing NCCL. Bring down unused ports with `sudo ip link set <iface> down`
3. `NCCL using network Socket` instead of IB — `NCCL_IB_DISABLE=1` was set (IB check failed on that node). Check that the IB interface has carrier=1 on both nodes

#### Compile Cache
The triton compile cache lives at `/tmp/torchinductor_root` inside the container.
Without a persistent volume, every container restart recompiles from scratch (~5 min).
Mount it to the host to persist across restarts:
```bash
export VLLM_SPARK_EXTRA_DOCKER_ARGS="-v /tmp/vllm-compile-cache:/tmp/torchinductor_root"
```

### Stop Cluster
```bash
./launch-cluster.sh stop
```

### Check Status
```bash
curl http://192.168.0.27:8000/health
./launch-cluster.sh status
```
