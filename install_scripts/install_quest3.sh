#!/usr/bin/env bash
# install_quest3.sh
# Sets up the gear_sonic_teleop venv for Quest 3 VR teleop.
# Same as install_pico.sh but WITHOUT XRoboToolkit SDK, plus the
# 'websockets' package for the WebXR bridge.
#
# Usage:  bash install_scripts/install_quest3.sh   (run from repo root)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ARCH="$(uname -m)"
echo "[OK] Architecture: $ARCH"

# ── 1. Ensure uv is installed ────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "[INFO] uv not found – installing via official installer …"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    if [ -f "$HOME/.local/bin/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.local/bin/env"
    elif [ -f "$HOME/.cargo/env" ]; then
        # shellcheck disable=SC1091
        source "$HOME/.cargo/env"
    else
        export PATH="$HOME/.local/bin:$PATH"
    fi

    if ! command -v uv &>/dev/null; then
        echo "[ERROR] uv installation succeeded but binary not found on PATH."
        echo "        Please add ~/.local/bin (or ~/.cargo/bin) to your PATH and re-run."
        exit 1
    fi
fi
echo "[OK] uv $(uv --version)"

# ── 2. Install uv-managed Python 3.10 ────────────────────────────────────────
echo "[INFO] Installing uv-managed Python 3.10 …"
uv python install 3.10
MANAGED_PY="$(uv python find --no-project 3.10)"
echo "[OK] Using Python: $MANAGED_PY"

# ── 3. Clean previous venv ───────────────────────────────────────────────────
cd "$REPO_ROOT"
echo "[INFO] Removing old .venv_teleop (if present) …"
rm -rf .venv_teleop

# ── 4. Create venv & install teleop extra + websockets ────────────────────────
echo "[INFO] Creating .venv_teleop with uv-managed Python 3.10 …"
uv venv .venv_teleop --python "$MANAGED_PY" --prompt gear_sonic_teleop
# shellcheck disable=SC1091
source .venv_teleop/bin/activate

echo "[INFO] Installing gear_sonic[teleop] …"
uv pip install -e "gear_sonic[teleop]"

echo "[INFO] Installing websockets (for Quest 3 WebXR bridge) …"
uv pip install websockets

# ── 5. Generate self-signed TLS certificate ───────────────────────────────────
CERT_DIR="$REPO_ROOT/gear_sonic/utils/teleop/vr/quest3_certs"
CERT_FILE="$CERT_DIR/cert.pem"
KEY_FILE="$CERT_DIR/key.pem"

if [ ! -f "$CERT_FILE" ] || [ ! -f "$KEY_FILE" ]; then
    echo "[INFO] Generating self-signed TLS certificate for WebXR …"
    mkdir -p "$CERT_DIR"
    openssl req -x509 -newkey rsa:2048 \
        -keyout "$KEY_FILE" -out "$CERT_FILE" \
        -days 365 -nodes \
        -subj "/CN=quest3-teleop"
    echo "[OK] Certificate generated: $CERT_FILE"
else
    echo "[OK] TLS certificate already exists: $CERT_FILE"
fi

# ── 6. sim extra + unitree_sdk2_python (desktop only) ─────────────────────────
if [ "$ARCH" = "aarch64" ] && [ "$(whoami)" = "unitree" ]; then
    echo "[SKIP] Skipping sim extra & unitree_sdk2_python (onboard Jetson Orin)"
else
    echo "[INFO] Installing sim extra …"
    uv pip install -e "gear_sonic[sim]"

    echo "[INFO] Installing unitree_sdk2_python …"
    uv pip install -e external_dependencies/unitree_sdk2_python
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Quest 3 teleop setup complete!"
echo ""
echo "  Activate the venv:"
echo "    source .venv_teleop/bin/activate"
echo ""
echo "  Run the Quest 3 manager:"
echo "    ./gear_sonic/scripts/sim_onnx_planner.sh --vr        # X2 planner stack + Quest manager (see docs/x2/F02_quest_vr.md)"
echo ""
echo "  Then open https://<workstation-ip>:8443 in the Quest 3"
echo "  browser to connect."
echo "══════════════════════════════════════════════════════════════"
