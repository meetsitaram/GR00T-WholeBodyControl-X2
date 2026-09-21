#!/bin/bash
# Install/refresh the X2 camera bridge as a boot service ON PC2. Run on PC2 as user `run`
# (passwordless sudo). Idempotent. Stops any hand-started bridge first.
#   bash /home/run/gear-sonic/x2_pc2/install_camera_bridge_service.sh
set -e
cd /home/run/gear-sonic
sudo cp x2_pc2/x2-camera-bridge.service /etc/systemd/system/x2-camera-bridge.service
sudo systemctl daemon-reload
for p in $(pgrep -f 'camera_zmq_[p]ublisher.py'); do sudo kill $p 2>/dev/null || true; done
sudo systemctl enable --now x2-camera-bridge.service
sleep 6
systemctl is-active x2-camera-bridge.service
tail -n 2 /home/run/gear-sonic/log/camera_zmq.log
