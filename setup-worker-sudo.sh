#!/bin/bash
# setup-worker-sudo.sh - Configure passwordless sudo for weiss user on worker node
# Run on WORKER node (192.168.0.176)
# Required because launch script runs sysctl, nvidia-smi, drop_caches inside container

echo "Configuring passwordless sudo for weiss user..."

cat <<EOF | sudo tee /etc/sudoers.d/weiss
weiss ALL=(ALL) NOPASSWD: ALL
EOF

sudo chmod 440 /etc/sudoers.d/weiss
sudo visudo -c

echo "Done. Test with: ssh weiss@192.168.0.176 'sudo whoami'"
