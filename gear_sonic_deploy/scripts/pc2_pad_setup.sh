#!/bin/bash
# One-time gamepad (Xbox-class controller) enablement on the robot's PC2.
#
# Captured 2026-07-28 from a working robot: three pieces make the pad chain
# work with the daemon-owned pad bridge (x2_pc2_daemons.sh session
# ``pad_bridge`` running pad_locomotion_bridge.py):
#
#   1. xpadneo — kernel driver for Xbox controllers over Bluetooth
#      (stock xpad does not expose the BT gamepad reliably; hid_xpadneo does).
#   2. hidraw permissions — the bridge reads /dev/hidraw* as a non-root user.
#   3. bluez policy — auto-enable the adapter on boot and allow seamless
#      re-pairing after the controller was paired elsewhere
#      (JustWorksRepairing), otherwise every re-pair needs a console.
#
# Run ON PC2 as the run user (needs sudo). Idempotent.

set -euo pipefail

echo "[pad-setup] 1/3 xpadneo driver"
if ! modinfo hid_xpadneo >/dev/null 2>&1; then
  sudo apt-get update && sudo apt-get install -y dkms git
  git clone https://github.com/atar-axis/xpadneo.git /tmp/xpadneo
  (cd /tmp/xpadneo && sudo ./install.sh)
else
  echo "  hid_xpadneo already installed"
fi

echo "[pad-setup] 2/3 hidraw udev rule"
echo 'KERNEL=="hidraw*", MODE="0666"' | sudo tee /etc/udev/rules.d/90-hidraw.rules >/dev/null
sudo udevadm control --reload-rules && sudo udevadm trigger

echo "[pad-setup] 3/3 bluez policy"
sudo python3 - << 'EOF'
import configparser, io
p = "/etc/bluetooth/main.conf"
c = configparser.ConfigParser(); c.optionxform = str
c.read(p)
for sect, key, val in [("General", "JustWorksRepairing", "always"),
                       ("General", "Privacy", "device"),
                       ("Policy", "AutoEnable", "true")]:
    if not c.has_section(sect):
        c.add_section(sect)
    c.set(sect, key, val)
buf = io.StringIO(); c.write(buf, space_around_delimiters=False)
open(p, "w").write(buf.getvalue())
print("  main.conf updated")
EOF
sudo systemctl restart bluetooth

echo "[pad-setup] done. Pair the controller once:"
echo "  bluetoothctl -- scan on ; pair <MAC> ; trust <MAC> ; connect <MAC>"
echo "Then x2_pc2_daemons.sh start brings up the pad bridge."
