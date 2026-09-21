#!/bin/bash
# Install/refresh the boot daemon that pairs the gamepad and fires the
# ignition ritual on the pad chord: x2_pc2/x2-pad-autopair.service ->
# x2_pad_autopair.py --start-cmd ritual_start_demo.sh. Run ON PC2 as the
# run user (passwordless sudo). Idempotent. Run pc2_pad_setup.sh (xpadneo,
# hidraw, bluez policy) once before this.
#   bash ${PC2_PREFIX:-/home/run/gear-sonic}/x2_pc2/install_pad_autopair_service.sh
set -e
PREFIX="${PC2_PREFIX:-/home/run/gear-sonic}"
UNIT="${PREFIX}/x2_pc2/x2-pad-autopair.service"
[ -f "$UNIT" ] || { echo "missing $UNIT (pc2_bringup.sh step 7b stages it)" >&2; exit 1; }
for f in x2_pad_autopair.py pc2_pad_daemon.py ritual_start_demo.sh; do
    [ -f "${PREFIX}/$f" ] || echo "WARNING: ${PREFIX}/$f missing -- push it (x2_pc2/push_to_pc2.sh) before the first chord" >&2
done
# The unit ships with the default prefix baked in; rewrite for a relocated install.
sed "s#/home/run/gear-sonic#${PREFIX}#g" "$UNIT" | sudo tee /etc/systemd/system/x2-pad-autopair.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now x2-pad-autopair.service
sleep 3
systemctl is-active x2-pad-autopair.service
tail -n 3 "${PREFIX}/log/pad_autopair.log" 2>/dev/null || true
