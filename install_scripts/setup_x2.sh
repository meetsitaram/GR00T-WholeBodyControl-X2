#!/usr/bin/env bash
# setup_x2.sh — zero-friction setup for the X2 SONIC sim + deploy stacks.
#
# One command from a fresh clone to a working install:
#
#     git clone <repo> && cd <repo> && bash install_scripts/setup_x2.sh
#
# What it does (idempotent — safe to re-run any time):
#   1. git lfs install + git lfs pull        (motion PKLs / MJCF meshes)
#   2. python3.10 venv at <repo>/.venv       (skipped if it already exists)
#   3. validated pip sequence (see requirements-x2.txt): CPU torch wheel FIRST, then
#      gear_sonic[sim], onnxruntime, pygame, websockets, motionbricks, huggingface_hub[cli]
#   4. models: the SONIC policy set SHIPS in-repo (gear_sonic_deploy/models/,
#      git-lfs) -- no download needed, only `git lfs pull` on a fresh clone.
#      Optional extras (planner graphs / other sets) come from
#      `download_from_hf.py --robot x2` when $SONIC_X2_HF_REPO is set
#      (bring-your-own, see MODELS.md)
#   5. optional: --with-docker builds the gear_sonic_deploy/docker_x2 sim
#      deploy image (skipped gracefully when docker is absent)
#   6. verification: package imports + model paths + next steps
#
# Model cache layout (multi-embodiment):
#   $SONIC_HOME             root, default ~/.cache/sonic
#   $SONIC_HOME/x2          X2 artifacts (sonic_policy/ + kplanner_onnx/
#                           layout, see MODELS.md); this subtree alone is
#                           overridable via $SONIC_X2_MODELS
#   $SONIC_HOME/g1          G1 artifacts (nvidia/GEAR-SONIC, via
#                           `download_from_hf.py --robot g1`)
# The stack scripts resolve models automatically from this cache; flags
# are only needed to override.
#
# Usage:
#   bash install_scripts/setup_x2.sh [--with-docker] [--skip-models]
#
# Env overrides:
#   SONIC_X2_HF_REPO   OPTIONAL Hub repo id for extra X2 artifacts (planner
#                      graphs, other policy sets); the shipped SONIC set
#                      needs no download
#   X2_MOTION_PKL      motion-lib PKL used by the LFS check and the quickstart
#                      hint (default: gear_sonic/data/motions/x2_demo_bank.pkl,
#                      built by tools/build_demo_bank_from_upstream.sh)

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SONIC_HOME="${SONIC_HOME:-$HOME/.cache/sonic}"
SONIC_X2_MODELS="${SONIC_X2_MODELS:-$SONIC_HOME/x2}"
X2_MOTION_PKL="${X2_MOTION_PKL:-gear_sonic/data/motions/x2_demo_bank.pkl}"
VENV="${REPO_ROOT}/.venv"
PIP="${VENV}/bin/pip"
PY="${VENV}/bin/python"

WITH_DOCKER=0
SKIP_MODELS=0
for arg in "$@"; do
    case "${arg}" in
        --with-docker) WITH_DOCKER=1 ;;
        --skip-models) SKIP_MODELS=1 ;;
        -h|--help)
            sed -n '2,38p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "unknown arg: ${arg} (try --help)" >&2; exit 2 ;;
    esac
done

C_GREEN=$'\e[32m'; C_YELLOW=$'\e[33m'; C_RED=$'\e[31m'; C_RESET=$'\e[0m'
step() { printf '\n%s==> %s%s\n' "${C_GREEN}" "$*" "${C_RESET}"; }
note() { printf '%s    %s%s\n' "${C_YELLOW}" "$*" "${C_RESET}"; }
fail() { printf '%sERROR: %s%s\n' "${C_RED}" "$*" "${C_RESET}" >&2; exit 1; }

cd "${REPO_ROOT}"

# ---------------------------------------------------------------- 1. git lfs
step "1/6 git lfs (motion PKLs, MJCF meshes are LFS-tracked)"
if ! command -v git-lfs >/dev/null 2>&1; then
    fail "git-lfs is not installed. Install it (e.g. 'sudo apt install git-lfs') and re-run."
fi
git lfs install || fail "git lfs install failed"
git lfs pull || fail "git lfs pull failed"
if [[ -f "${X2_MOTION_PKL}" ]]; then
    note "lfs OK ($(ls -la "${X2_MOTION_PKL}" | awk '{print $5}') bytes in ${X2_MOTION_PKL} — a pointer stub would be ~130)"
else
    note "${X2_MOTION_PKL} not present — build it with tools/build_demo_bank_from_upstream.sh (or set X2_MOTION_PKL)"
fi

# ------------------------------------------------------------------- 2. venv
step "2/6 python3.10 venv at ${VENV}"
if [[ -x "${PY}" ]]; then
    note "venv already exists — skipping creation"
else
    command -v python3.10 >/dev/null 2>&1 || fail "python3.10 not found (the validated interpreter — see requirements-x2.txt)"
    python3.10 -m venv "${VENV}" || fail "venv creation failed"
fi
"${PIP}" install --upgrade pip || fail "pip upgrade failed"

# ----------------------------------------------------------------- 3. pip seq
# Order matters: CPU torch wheel FIRST so gear_sonic's torch>=2.4.0 dep is
# already satisfied and pip does not pull the multi-GB CUDA wheel.
step "3/6 pip install (validated sequence from requirements-x2.txt)"
"${PIP}" install torch --index-url https://download.pytorch.org/whl/cpu || fail "torch (cpu) install failed"
"${PIP}" install -e "./gear_sonic[sim]" || fail "gear_sonic[sim] install failed"
"${PIP}" install onnxruntime pygame websockets || fail "onnxruntime/pygame/websockets install failed"
"${PIP}" install -e ./motionbricks || fail "motionbricks install failed"
"${PIP}" install -U "huggingface_hub[cli]" || fail "huggingface_hub install failed"

# ----------------------------------------------------------------- 4. models
step "4/6 X2 models: shipped set (gear_sonic_deploy/models/) + optional cache (${SONIC_X2_MODELS})"
# The SONIC policy set ships in-repo via git-lfs (gear_sonic_deploy/models/
# README.md); the launchers pick it automatically. A clone without git-lfs
# holds ~130-byte pointer files instead -- detect and say so.
SHIPPED_MODELS_DIR="${REPO_ROOT}/gear_sonic_deploy/models"
SHIPPED_FILES=(
    "x2_sonic_v16ft8_45000_dual.onnx"
    "x2_sonic_v16ft8_45000_g1.onnx"
    "x2_sonic_v16ft8_45000_g1_token.onnx"
    "x2_sonic_v16ft8_45000_smpl_tokenizer.onnx"
)
shipped_missing=0
for f in "${SHIPPED_FILES[@]}"; do
    if [[ ! -f "${SHIPPED_MODELS_DIR}/${f}" ]] || head -c 40 "${SHIPPED_MODELS_DIR}/${f}" | grep -q 'git-lfs'; then
        shipped_missing=1
    fi
done
if [[ "${shipped_missing}" -eq 0 ]]; then
    note "shipped SONIC set present (gear_sonic_deploy/models/x2_sonic_v16ft8_45000_*) — no download needed"
else
    note "shipped SONIC set is missing or LFS pointers — run: git lfs install && git lfs pull"
fi
# Optional cache (planner graphs x2_kplanner_{template,velocity}.onnx and any
# other policy set; MODELS.md): filled by download_from_hf.py --robot x2 from
# $SONIC_X2_HF_REPO when you have one, otherwise bring-your-own.
KEY_FILES=(
    "kplanner_onnx/x2_kplanner_template.onnx"
    "kplanner_onnx/x2_kplanner_velocity.onnx"
)
missing=0
for f in "${KEY_FILES[@]}"; do
    [[ -f "${SONIC_X2_MODELS}/${f}" ]] || missing=1
done
if [[ "${SKIP_MODELS}" -eq 1 ]]; then
    note "--skip-models: cache download skipped"
elif [[ "${missing}" -eq 0 ]]; then
    note "planner graphs already present in the cache — skipping download"
elif [[ -z "${SONIC_X2_HF_REPO:-}" ]]; then
    note "SONIC_X2_HF_REPO is not set — nothing to download (the shipped SONIC set needs none)."
    note "Planner graphs are bring-your-own (MODELS.md, PLANNER_MODEL row): export them or"
    note "place them under ${SONIC_X2_MODELS}/kplanner_onnx/; SONIC_X2_HF_REPO=<hub repo id> is an optional override."
else
    note "if the download fails with 401/403 (private repo), run 'hf auth login' first."
    SONIC_HOME="${SONIC_HOME}" SONIC_X2_MODELS="${SONIC_X2_MODELS}" \
        "${PY}" "${REPO_ROOT}/download_from_hf.py" --robot x2 \
        || fail "model download failed (private repo? run: ${VENV}/bin/hf auth login)"
fi

# --------------------------------------------------- 4b. X2 vendor meshes
step "4b/6 X2 robot meshes (AgiBot official URDF package)"
X2_URDF_URL="${X2_URDF_URL:-https://x2-aimdk.agibot.com/en/latest/_downloads/2ffc9785259556f409e385974a7a0461/X2_URDF-v1.3.0.zip}"
MESH_DEST="${REPO_ROOT}/gear_sonic/data/assets/robot_description/urdf/x2_ultra/meshes"
if compgen -G "${MESH_DEST}/*.STL" > /dev/null; then
    note "meshes already present — skipping download"
else
    note "fetching AgiBot X2 URDF package (~50 MB; X2_URDF_URL to override)"
    _tmp="$(mktemp -d)"
    if curl -fSL --retry 2 -o "${_tmp}/x2_urdf.zip" "${X2_URDF_URL}" \
       && (cd "${_tmp}" && unzip -q x2_urdf.zip "*/meshes/*"); then
        find "${_tmp}" -type f \( -name "*.STL" -o -name "*.stl" \) -exec cp {} "${MESH_DEST}/" \;
        note "installed $(ls "${MESH_DEST}" | grep -ci stl) meshes"
    else
        note "mesh download FAILED — sim/viewer needs the meshes."
        note "If the direct URL moved, find the current URDF package link on:"
        note "  https://x2-aimdk.agibot.com/en/latest/get_sdk/index.html"
        note "Manual fallback in ${MESH_DEST}/README.md"
    fi
    rm -rf "${_tmp}"
fi

# ------------------------------------------------ 4c. OmniHand stock meshes
step "4c/6 OmniHand meshes (AgiBot official URDF package)"
OMNIHAND_URDF_URL="${OMNIHAND_URDF_URL:-https://www.agibot.com.cn/file/ueditor/php/upload/file/20260201/1769933923166393.zip}"
OH_DEST="${REPO_ROOT}/gear_sonic/data/assets/robot_description/omnihand/meshes"
if compgen -G "${OH_DEST}/finger_*.STL" > /dev/null; then
    note "OmniHand meshes already present — skipping download"
else
    note "fetching AgiBot OmniHand 2025 URDF package (~6 MB; OMNIHAND_URDF_URL to override)"
    note "  (if the URL moved, current links: https://www.agibot.com.cn/DOCS/OS/Omnihand-O10)"
    _tmp="$(mktemp -d)"
    if curl -fSL --retry 2 -o "${_tmp}/oh.zip" "${OMNIHAND_URDF_URL}" \
       && (cd "${_tmp}" && unzip -q oh.zip); then
        find "${_tmp}" -type f \( -name "*.STL" -o -name "*.stl" \) -exec cp {} "${OH_DEST}/" \;
        note "installed $(ls "${OH_DEST}" | grep -ci stl) OmniHand meshes (2 custom clipped ones ship in-repo)"
    else
        note "OmniHand mesh download FAILED — hands render without visuals; see ${OH_DEST}/README.md"
    fi
    rm -rf "${_tmp}"
fi

# ----------------------------------------------------------------- 5. docker
step "5/6 docker sim deploy image (optional)"
if [[ "${WITH_DOCKER}" -eq 1 ]]; then
    # The deploy image builds against AgiBot's aimdk_msgs, which AgiBot
    # distributes with their AimDK SDK (no direct-download URL, so this
    # script cannot fetch it). Check EARLY with clear instructions instead
    # of failing 10 minutes into the docker build.
    AIMDK_SDK_URL="${AIMDK_SDK_URL:-https://x2-aimdk.agibot.com/downloads/aimdk-aarch64-a424add7-artifacts.zip}"
    AIMDK_DEST="${REPO_ROOT}/gear_sonic_deploy/thirdparty/aimdk_msgs"
    if [[ ! -f "${AIMDK_DEST}/package.xml" ]]; then
        note "aimdk_msgs not present — fetching AgiBot's official AimDK SDK artifact"
        note "  (${AIMDK_SDK_URL} — override via AIMDK_SDK_URL;"
        note "   if the URL moved, current links live on"
        note "   https://x2-aimdk.agibot.com/en/latest/about_agibot_X2/robot_specifications.html"
        note "   and https://x2-aimdk.agibot.com/en/latest/get_sdk/index.html)"
        _tmp="$(mktemp -d)"
        if curl -fSL --retry 2 -o "${_tmp}/aimdk.zip" "${AIMDK_SDK_URL}" \
           && (cd "${_tmp}" && unzip -q aimdk.zip "*/src/aimdk_msgs/*" -x "*/prebuilt_*/*") ; then
            _src="$(find "${_tmp}" -type d -path "*/src/aimdk_msgs" | head -1)"
            if [[ -n "${_src}" && -f "${_src}/package.xml" ]]; then
                cp -r "${_src}/." "${AIMDK_DEST}/"
                note "aimdk_msgs installed from the official SDK artifact"
            fi
        fi
        rm -rf "${_tmp}"
    fi
    if [[ ! -f "${AIMDK_DEST}/package.xml" ]]; then
        note "aimdk_msgs auto-download failed — skipping the docker build."
        note "Manual one-time step: download the AimDK SDK from"
        note "  https://x2-aimdk.agibot.com/en/latest/get_sdk/index.html"
        note "then: cp -r <sdk>/src/aimdk_msgs/* ${AIMDK_DEST}/"
        note "and re-run: bash install_scripts/setup_x2.sh --with-docker"
    elif command -v docker >/dev/null 2>&1; then
        # Same build enter_sim.sh runs (idempotent; ~10 min first time).
        (cd "${REPO_ROOT}/gear_sonic_deploy/docker_x2" && docker compose build) \
            || fail "docker image build failed (see gear_sonic_deploy/docker_x2/README.md)"
        note "docker_x2 image built"
    else
        note "docker not found — skipping image build (install docker, then re-run with --with-docker)"
    fi
else
    note "skipped (pass --with-docker to build the sim deploy image; only needed for the full deploy loop)"
fi

# ----------------------------------------------------------- 6. verification
step "6/6 verification"
"${PY}" -c "import mujoco, onnxruntime, gear_sonic; print('imports OK: mujoco', mujoco.__version__, '| onnxruntime', onnxruntime.__version__, '| gear_sonic')" \
    || fail "import check failed"
echo
echo "shipped SONIC set (gear_sonic_deploy/models/):"
for f in "${SHIPPED_FILES[@]}"; do
    if [[ -f "${SHIPPED_MODELS_DIR}/${f}" ]] && ! head -c 40 "${SHIPPED_MODELS_DIR}/${f}" | grep -q 'git-lfs'; then
        printf '  %-45s %s\n' "${f}" "$(du -h "${SHIPPED_MODELS_DIR}/${f}" | cut -f1)"
    else
        printf '  %-45s %sMISSING / LFS pointer (git lfs pull)%s\n' "${f}" "${C_RED}" "${C_RESET}"
    fi
done
echo "model cache (${SONIC_X2_MODELS}, optional):"
for f in "${KEY_FILES[@]}"; do
    if [[ -f "${SONIC_X2_MODELS}/${f}" ]]; then
        printf '  %-45s %s\n' "${f}" "$(du -h "${SONIC_X2_MODELS}/${f}" | cut -f1)"
    else
        printf '  %-45s %sMISSING%s\n' "${f}" "${C_RED}" "${C_RESET}"
    fi
done

cat <<EOF

${C_GREEN}Setup complete.${C_RESET} Next steps (see README.md / MODELS.md):

  # whole-body sim rehearsal on the shipped set, no model flags needed:
  ./gear_sonic/scripts/sim_onnx_planner.sh --whole-body-teleop

  # headless policy eval — model auto-found, no flags needed:
  .venv/bin/python gear_sonic/scripts/eval_x2_mujoco_onnx.py \\
      --motion ${X2_MOTION_PKL} --no-viewer --total-sim-seconds 20

  # direct PKL playback stack in sim:
  bash gear_sonic/scripts/run_x2_pkl_direct_stack.sh

  # Quest 3 VR teleop stack preflight:
  bash gear_sonic/scripts/run_x2_quest3_planner_stack.sh --validate-only --no-x2-debug-bridge
EOF
