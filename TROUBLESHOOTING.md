# Troubleshooting Guide

## SSH Daemon Unavailability - March 23, 2026

### Symptoms
- SSH connections dropped unexpectedly
- System became unresponsive to network requests
- Required cold reset to recover

### Root Cause Analysis

#### Issue 1: Out of Memory (OOM)

The system ran out of memory due to unrestricted Docker container memory usage.

**Timeline (Mar 23, 2026):**
| Time | Event |
|------|-------|
| ~20:17 | OOM killer first triggered |
| 20:17-21:04 | Continuous OOM events (47 minutes of memory thrashing) |
| 21:04:07 | Critical system services killed (polkitd, rsyslogd, gdm3, etc.) |
| 21:04:07 | SSH PAM session forcibly closed |
| 21:47:xx | Cold reset / reboot |

**The culprit:** `VLLM::EngineCore` process consuming massive virtual memory (~154 GB virtual, significant physical RAM for model weights and KV cache) with no Docker memory limits.

**Services killed by OOM:**
- `rsyslogd` → logging stopped
- `polkitd` → authentication degraded
- `gdm3`, `Xorg`, `mutter` → GUI collapsed
- `pipewire`, `wireplumber` → audio died
- `systemd-resolve` → DNS failed
- SSH sessions terminated by PAM

#### Issue 2: System Suspend

On subsequent boot, GNOME's power manager (`gsd-power`) attempted to suspend the system:

```
Mar 23 22:03:58 gx10-ccb8 gsd-power[2423]: Error calling suspend action: GDBus.Error:org.freedesktop.DBus.Error.AccessDenied: Permission denied
```

The suspend was blocked by permissions but indicated the system was trying to sleep, which would kill SSH connections.

### Fixes Applied

#### 1. Docker Memory Limits

Added to `docker-compose-qwen35-nvfp4.yaml`:

```yaml
mem_limit: 90g
memswap_limit: 110g
```

This ensures the container cannot consume more than 90GB RAM, leaving ~30GB for the OS and other services.

#### 2. Reduced Context Length

Changed from 262144 (256k) to 131072 (128k) tokens:
- Reduces KV cache memory requirements by ~50%
- Still sufficient for most use cases
- More stable operation on single DGX Spark

#### 3. Reduced GPU Memory Utilization

Set `--gpu-memory-utilization 0.82` (was 0.87-0.90) to leave headroom.

#### 4. System Suspend Prevention (requires sudo)

Create `/etc/systemd/logind.conf.d/disable-suspend.conf`:

```bash
sudo mkdir -p /etc/systemd/logind.conf.d
sudo tee /etc/systemd/logind.conf.d/disable-suspend.conf << 'EOF'
[Login]
HandlePowerKey=ignore
HandleSuspendKey=ignore
HandleHibernateKey=ignore
HandleLidSwitch=ignore
HandleLidSwitchExternalPower=ignore
IdleAction=ignore
EOF
sudo systemctl restart systemd-logind
```

This prevents systemd-logind from responding to any suspend/hibernate events, overriding GNOME's power management.

### Current Stable Configuration

```yaml
# docker-compose-qwen35-nvfp4.yaml (key settings)
services:
  vllm-qwen35-nvfp4:
    shm_size: 48g
    mem_limit: 90g
    memswap_limit: 110g
    command:
      - --max-model-len 131072
      - --gpu-memory-utilization 0.9
      - --kv-cache-dtype fp8
      - --max-num-seqs 32
```

### Lessons Learned

1. **Always set Docker memory limits** for large ML workloads - containers can otherwise consume all host RAM
2. **Server systems should disable suspend** at the systemd level, not just in GNOME settings
3. **Monitor memory usage** during initial model loading - the 122B model with 256k context was too aggressive for single Spark
4. **OOM kills cascade** - once rsyslogd dies, you lose visibility into what's happening

### Monitoring Commands

```bash
# Check container memory usage
docker stats --no-stream

# Check system memory
free -h

# Check for OOM events in logs
journalctl -k | grep -i "out of memory"

# Check for suspend attempts
journalctl | grep -i "suspend\|sleep" | grep -v "Sleep Button"

# Health check
curl -s http://localhost:8000/health
```

### Quick Recovery

If the system becomes unresponsive:

1. Check if OOM is occurring: `dmesg | tail -50`
2. If container is the cause: `docker stop vllm-qwen35-nvfp4`
3. Check memory: `free -h`
4. Restart with lower settings if needed

---

## IB/RoCE Networking - April 5, 2026

### Symptoms
- `NCCL_IB : No device found` — IB not detected
- `ibv_modify_qp failed with 22 Invalid argument` — QP creation fails
- `NCCL using network Socket` — fell back to TCP instead of IB
- Worker crashes during `ncclCommInitRank`

### Root Causes Found

#### 1. Phantom Carrier on Secondary NIC
The DGX Spark has two CX-7 NICs. The second (`enP2p1s0f*`) shows `carrier=1` even when unplugged. NCCL discovers it and tries to use it, causing QP failures.

**Fix:** Bring down the phantom interface on both nodes after every reboot:
```bash
sudo ip link set enP2p1s0f0np0 down  # on both nodes
```

#### 2. Cross-Port Cabling
If the IB cable connects different port numbers on each Spark (e.g., head `enP2p1s0f1np1` ↔ worker `enp1s0f0np0`), the RoCE GID tables have IPv4 GIDs at different indices. NCCL picks mismatched GIDs and `ibv_modify_qp` fails with EINVAL.

**Fix:** Cable matching ports: head `enp1s0f0np0` ↔ worker `enp1s0f0np0`

#### 3. Container Has No `ping`
The IB link check in the launch script used `ping`, which doesn't exist in the container. The check silently failed and disabled IB.

**Fix:** Use sysfs carrier detect instead: `cat /sys/class/net/$iface/carrier`

#### 4. RDMA Socket Buffers Too Small
Default `net.core.rmem_max` = 212992 bytes (~208KB). NCCL over RoCE needs ≥16MB.

**Fix:** Set in the launch script (runs inside privileged container):
```bash
sysctl -w net.core.rmem_max=16777216
sysctl -w net.core.wmem_max=16777216
```

#### 5. IB_IF in .env Causes Device Mismatch
Setting `IB_IF=rocep1s0f0` in `.env` sets `NCCL_IB_HCA` to that exact device name. If the device doesn't exist on a node (different hardware naming), NCCL fails.

**Fix:** Leave `IB_IF` unset in `.env`. Let NCCL auto-discover active RoCE devices.

#### 6. GPU Memory OOM After Compilation
With `--gpu-memory-utilization 0.80`, the model loads fine but fails during KV cache allocation after torch.compile. Compiled CUDA graphs are resident in GPU memory and need more headroom.

**Fix:** Set `--gpu-memory-utilization 0.92` (from 0.80). The 397B INT4 model + compiled graphs + fp8 KV cache fits in ~98GB of 128GB.

### Verification

```bash
# Check only correct IB port is active
cat /sys/class/net/enp1s0f0np0/carrier   # should be 1
cat /sys/class/net/enP2p1s0f0np0/carrier # should be 0 (or empty)

# Verify IB IP connectivity
ping -c 1 -W 2 10.10.10.6  # from head
ping -c 1 -W 2 10.10.10.5  # from worker

# Check NCCL uses IB in logs
docker logs vllm_node 2>&1 | grep "Using network IB"
```
