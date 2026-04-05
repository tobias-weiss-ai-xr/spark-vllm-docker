#!/bin/bash
# setup-ib.sh - Configure IB interfaces for cluster communication
# Run on HEAD node BEFORE launch-cluster.sh
# Requires passwordless SSH to worker node

WORKER_IP="192.168.0.176"
HEAD_IB_IP="10.10.10.5"
WORKER_IB_IP="10.10.10.6"
HEAD_IB_IF="enp1s0f0np0"
WORKER_IB_IF="enp1s0f0np0"

echo "=== IB Interface Setup ==="

# Head node
echo "Configuring head IB interface ($HEAD_IB_IF)..."
ip addr flush dev $HEAD_IB_IF 2>/dev/null || true
ip addr add $HEAD_IB_IP/30 dev $HEAD_IB_IF
ip link set $HEAD_IB_IF mtu 9000 up
echo "  $HEAD_IB_IF: $HEAD_IB_IP/30"

# Worker node
echo "Configuring worker IB interface ($WORKER_IB_IF)..."
ssh -o BatchMode=yes -o StrictHostKeyChecking=no weiss@$WORKER_IP "
    sudo ip addr flush dev $WORKER_IB_IF 2>/dev/null || true
    sudo ip addr add $WORKER_IB_IP/30 dev $WORKER_IB_IF
    sudo ip link set $WORKER_IB_IF mtu 9000 up
    echo '  $WORKER_IB_IF: $WORKER_IB_IP/30'
"

sleep 1

# Verify
if ping -c 1 -W 2 $WORKER_IB_IP &>/dev/null; then
    echo "IB link verified: $HEAD_IB_IP <-> $WORKER_IP"
else
    echo "ERROR: IB link not responding. Check cable connection."
    exit 1
fi
