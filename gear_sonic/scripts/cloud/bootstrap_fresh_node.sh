#!/usr/bin/env bash
# Bootstrap a fresh Nebius (or any Ubuntu 24.04 + CUDA 12) GPU node into a
# state where ``run_smoke_8gpu.sh`` can launch X2 Ultra training.
#
# Idempotent — every step checks for a sentinel and skips if already done, so
# re-running on a partially-set-up node is safe.
#
# Mirrors every fix in Appendix B of ``docs/source/user_guide/train-on-cloud.md``:
#   B.6   Conda ToS prompt
#   B.7   Pin isaacsim==5.1.0.0
#   B.8   Pre-accept Omniverse EULA env vars
#   B.9   setuptools 82 / flatdict / --no-build-isolation
#   B.10b open3d / tensordict / vector-quantize-pytorch missing extras
#   B.10c libglu1-mesa missing
#   B.10d NVIDIA Vulkan ICD blocked by libnvidia-gl apt pin
#   B.11  git-lfs for robot meshes
#
# What this script does NOT do:
#   - Install the NVIDIA driver or CUDA toolkit (assumes the boot image
#     already shipped them — Nebius ``ubuntu24.04-cuda12`` does).
#   - Pull the gitignored side-channel bundle (motion PKLs, sphere URDF).
#     That's an ``scp`` from your workstation, see §1 + §5 of the doc.
#   - Pull mesh-trained checkpoints. ``scp`` those separately if you want
#     to fine-tune (e.g. for the §11 sphere-feet experiment).
#
# Usage (on the cloud node, fresh ssh session):
#
#   wget https://raw.githubusercontent.com/<your-fork>/<branch>/gear_sonic/scripts/cloud/bootstrap_fresh_node.sh
#   # or scp it from your workstation:
#   #   scp gear_sonic/scripts/cloud/bootstrap_fresh_node.sh ubuntu@$IP:~/
#
#   chmod +x bootstrap_fresh_node.sh
#   REPO_URL=git@github.com:<your-fork>/GR00T-WholeBodyControl.git \
#     bash ~/bootstrap_fresh_node.sh 2>&1 | tee ~/bootstrap.log
#
# Override knobs (all optional):
#   REPO_URL         git URL to clone the GR00T repo from              (required if repo not yet cloned)
#   REPO_BRANCH      branch to check out                               (default: main)
#   REPO_DIR         where to clone it                                 (default: $HOME/GR00T-WholeBodyControl)
#   ISAACLAB_DIR     where to clone IsaacLab                           (default: $HOME/IsaacLab)
#   ISAACLAB_TAG     IsaacLab git tag                                  (default: v2.2.0)
#   ISAACSIM_VERSION isaacsim wheel version                            (default: 5.1.0.0)
#   CONDA_ENV        conda env name                                    (default: env_isaaclab)
#   PYTHON_VERSION   python version inside conda env                   (default: 3.11)
#   SKIP_VULKAN_FIX  set to 1 to skip the libnvidia-gl pin override   (default: unset)
#                    Use only if you know vulkaninfo already lists NVIDIA GPUs.

set -euo pipefail

REPO_URL=${REPO_URL:-}
REPO_BRANCH=${REPO_BRANCH:-main}
REPO_DIR=${REPO_DIR:-$HOME/GR00T-WholeBodyControl}
ISAACLAB_DIR=${ISAACLAB_DIR:-$HOME/IsaacLab}
ISAACLAB_TAG=${ISAACLAB_TAG:-v2.2.0}
ISAACSIM_VERSION=${ISAACSIM_VERSION:-5.1.0.0}
CONDA_ENV=${CONDA_ENV:-env_isaaclab}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}

CONDA_PREFIX_DIR=${CONDA_PREFIX_DIR:-$HOME/miniconda3}

log()  { printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
step() { printf '  - %s\n' "$*"; }
skip() { printf '  ~ skip: %s\n' "$*"; }

#-------------------------------------------------------------------------------
# Phase 0 — pre-flight
#-------------------------------------------------------------------------------
log "Phase 0: pre-flight"
if ! command -v nvidia-smi >/dev/null; then
  echo "FATAL: nvidia-smi not found. This script assumes the boot image already" >&2
  echo "       has the NVIDIA driver + CUDA toolkit (e.g. Nebius ubuntu24.04-cuda12)." >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader

#-------------------------------------------------------------------------------
# Phase 1 — OS packages (covers B.10c + B.11 prereq)
#-------------------------------------------------------------------------------
log "Phase 1: OS packages"
sudo apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  tmux htop rsync jq git git-lfs \
  libglu1-mesa vulkan-tools \
  build-essential ca-certificates curl wget

#-------------------------------------------------------------------------------
# Phase 2 — NVIDIA Vulkan ICD (B.10d, the BIGGEST gotcha)
#-------------------------------------------------------------------------------
log "Phase 2: NVIDIA Vulkan ICD (B.10d)"
if [[ "${SKIP_VULKAN_FIX:-0}" == "1" ]]; then
  skip "SKIP_VULKAN_FIX=1"
elif vulkaninfo --summary 2>/dev/null | grep -q "driverName.*NVIDIA"; then
  skip "vulkaninfo already lists NVIDIA driver"
else
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  DRV_MAJOR=${DRV%%.*}
  step "driver=${DRV} major=${DRV_MAJOR}"

  # Find a libnvidia-gl-${DRV_MAJOR} version that exactly matches the running driver.
  DRV_PIN=$(apt-cache madison "libnvidia-gl-${DRV_MAJOR}" 2>/dev/null \
    | awk -F'|' -v want="$DRV" 'index($2,want){gsub(/ /,"",$2); print $2; exit}' || true)

  APT_OK=0
  if [[ -n "${DRV_PIN}" ]]; then
    step "candidate version: ${DRV_PIN}"
    sudo tee /etc/apt/preferences.d/allow-libnvidia-gl >/dev/null <<EOF
Package: libnvidia-gl-${DRV_MAJOR}
Pin: version ${DRV_PIN}
Pin-Priority: 700
EOF
    sudo apt-get update -q
    if sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
        "libnvidia-gl-${DRV_MAJOR}=${DRV_PIN}"; then
      APT_OK=1
    fi
  fi

  # Fallback for compute-only images (e.g. Nebius ubuntu24.04-cuda13.0): the
  # dpkg driver subset has no GL/Vulkan userland and libnvidia-gl-*'s EGL deps
  # are not installable from any configured repo. Install the matching driver
  # runfile's USERSPACE over it (same version = same ABI; kernel modules kept).
  # Diagnosed 2026-08-06 on 8xH100: apt path dead-ends on libnvidia-egl-xcb1,
  # partial .deb extraction dies on a __malloc_hook symbol lookup in glcore —
  # the runfile build is the only self-consistent GL stack for these images.
  if [[ "${APT_OK}" -ne 1 ]]; then
    step "apt libnvidia-gl path unavailable — driver runfile userspace fallback"
    RUNFILE="NVIDIA-Linux-x86_64-${DRV}.run"
    ( cd /tmp \
      && { curl -sfO "https://download.nvidia.com/XFree86/Linux-x86_64/${DRV}/${RUNFILE}" \
           || curl -sfO "https://us.download.nvidia.com/XFree86/Linux-x86_64/${DRV}/${RUNFILE}"; } \
      && chmod +x "${RUNFILE}" \
      && sudo ./"${RUNFILE}" --silent --no-kernel-modules --no-questions \
           --ui=none --no-check-for-alternate-installs )
    sudo ldconfig
  fi

  # The runfile drops libraries the linker cache doesn't know about yet;
  # without this refresh the check below reports llvmpipe only and aborts a
  # perfectly good install (observed on node-1, 2026-08-07).
  sudo ldconfig

  step "post-install vulkaninfo:"
  vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverName" | head -20 \
    || echo "  (vulkaninfo still empty — investigate before continuing)"
  if ! vulkaninfo --summary 2>/dev/null | grep -q "NVIDIA"; then
    echo "FATAL: no NVIDIA Vulkan device after both install paths." >&2
    echo "       Try 'sudo ldconfig && vulkaninfo --summary' manually; if that" >&2
    echo "       lists the GPUs, re-run with SKIP_VULKAN_FIX=1." >&2
    exit 1
  fi
fi

#-------------------------------------------------------------------------------
# Phase 3 — Pre-accept Omniverse EULA env vars (B.8)
#-------------------------------------------------------------------------------
log "Phase 3: Omniverse EULA env vars (B.8)"
EULA_BLOCK_MARKER="# >>> isaac-eula-bootstrap >>>"
if grep -q "${EULA_BLOCK_MARKER}" "$HOME/.bashrc" 2>/dev/null; then
  skip "EULA env vars already in ~/.bashrc"
else
  cat >>"$HOME/.bashrc" <<'EOF'

# >>> isaac-eula-bootstrap >>>
# Pre-accept Omniverse / IsaacSim EULAs so non-tty installers don't hang.
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# <<< isaac-eula-bootstrap <<<
EOF
  step "appended EULA block to ~/.bashrc"
fi
# Apply to the current shell so the rest of this script benefits.
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#-------------------------------------------------------------------------------
# Phase 4 — Miniconda
#-------------------------------------------------------------------------------
log "Phase 4: Miniconda"
if [[ -d "${CONDA_PREFIX_DIR}" && -x "${CONDA_PREFIX_DIR}/bin/conda" ]]; then
  skip "${CONDA_PREFIX_DIR} already exists"
else
  step "downloading miniconda installer"
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "${CONDA_PREFIX_DIR}"
  "${CONDA_PREFIX_DIR}/bin/conda" init bash >/dev/null
fi

# shellcheck disable=SC1091
source "${CONDA_PREFIX_DIR}/etc/profile.d/conda.sh"

#-------------------------------------------------------------------------------
# Phase 5 — conda ToS (B.6)
#-------------------------------------------------------------------------------
log "Phase 5: conda ToS (B.6)"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true
step "ToS accept attempted for main + r channels"

#-------------------------------------------------------------------------------
# Phase 6 — env_isaaclab conda env
#-------------------------------------------------------------------------------
log "Phase 6: conda env '${CONDA_ENV}'"
if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  skip "env '${CONDA_ENV}' already exists"
else
  conda create -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${CONDA_ENV}"
python -V

#-------------------------------------------------------------------------------
# Phase 7 — setuptools / flatdict workaround (B.9)
#-------------------------------------------------------------------------------
log "Phase 7: setuptools + flatdict workaround (B.9)"
if python -c "import flatdict; assert flatdict.__version__ == '4.0.1'" 2>/dev/null; then
  skip "flatdict==4.0.1 already importable"
else
  pip install --quiet "setuptools==80.9.0" wheel
  pip install --quiet --no-build-isolation "flatdict==4.0.1"
  step "installed setuptools==80.9.0 + flatdict==4.0.1 (--no-build-isolation)"
fi

#-------------------------------------------------------------------------------
# Phase 8 — isaacsim wheel (B.7)
#-------------------------------------------------------------------------------
log "Phase 8: isaacsim==${ISAACSIM_VERSION} (B.7)"
if python -c "import isaacsim, sys; sys.exit(0)" 2>/dev/null; then
  step "isaacsim already importable; ensuring version pin"
  pip install --quiet "isaacsim[all,extscache]==${ISAACSIM_VERSION}"
else
  step "installing isaacsim[all,extscache]==${ISAACSIM_VERSION} (~865 MB torch wheel; takes 5-7 min)"
  pip install "isaacsim[all,extscache]==${ISAACSIM_VERSION}"
fi

#-------------------------------------------------------------------------------
# Phase 9 — IsaacLab git clone + ./isaaclab.sh -i (B.8)
#-------------------------------------------------------------------------------
log "Phase 9: IsaacLab ${ISAACLAB_TAG}"
if [[ -d "${ISAACLAB_DIR}/.git" ]]; then
  cur_tag=$(cd "${ISAACLAB_DIR}" && git describe --tags --always 2>/dev/null || echo "unknown")
  step "existing checkout at ${ISAACLAB_DIR} (${cur_tag})"
else
  step "cloning IsaacLab ${ISAACLAB_TAG} into ${ISAACLAB_DIR}"
  git clone --branch "${ISAACLAB_TAG}" --depth 1 \
    https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_DIR}"
fi

if python -c "import isaaclab" 2>/dev/null; then
  skip "isaaclab already importable"
else
  step "running isaaclab.sh -i (3-4 min)"
  (cd "${ISAACLAB_DIR}" && ./isaaclab.sh -i)
fi

#-------------------------------------------------------------------------------
# Phase 10 — git-lfs init (B.11)
#-------------------------------------------------------------------------------
log "Phase 10: git-lfs init (B.11)"
git lfs install --skip-repo
step "git-lfs hooks installed at user level"

#-------------------------------------------------------------------------------
# Phase 11 — Clone the GR00T repo + LFS pull robot meshes (B.11)
#-------------------------------------------------------------------------------
log "Phase 11: GR00T-WholeBodyControl repo"
if [[ -d "${REPO_DIR}/.git" ]]; then
  step "existing repo at ${REPO_DIR}; running git pull on '${REPO_BRANCH}'"
  (cd "${REPO_DIR}" && git fetch --all --quiet && git checkout "${REPO_BRANCH}" && git pull --ff-only)
else
  if [[ -z "${REPO_URL}" ]]; then
    echo "FATAL: REPO_DIR=${REPO_DIR} doesn't exist and REPO_URL is unset." >&2
    echo "       Re-run with: REPO_URL=git@github.com:<fork>/GR00T-WholeBodyControl.git $0" >&2
    exit 1
  fi
  step "cloning ${REPO_URL} (branch ${REPO_BRANCH}) into ${REPO_DIR}"
  git clone --branch "${REPO_BRANCH}" "${REPO_URL}" "${REPO_DIR}"
fi

step "git lfs pull (robot meshes, ~110 MB for x2_ultra)"
(cd "${REPO_DIR}" && git lfs pull --include "gear_sonic/data/assets/robot_description/**")

# Sanity-check a real mesh landed (not a 132-byte LFS pointer).
mesh_check="${REPO_DIR}/gear_sonic/data/assets/robot_description/urdf/x2_ultra/meshes/pelvis.STL"
if [[ -f "${mesh_check}" ]]; then
  size=$(stat -c%s "${mesh_check}")
  if [[ "${size}" -lt 1000 ]]; then
    echo "FATAL: ${mesh_check} is only ${size} bytes — LFS pull failed." >&2
    echo "       Try: cd ${REPO_DIR} && git lfs pull" >&2
    exit 1
  fi
  step "pelvis.STL = ${size} bytes (looks healthy)"
fi

#-------------------------------------------------------------------------------
# Phase 12 — gear_sonic[training] + B.10b extras
#-------------------------------------------------------------------------------
log "Phase 12: gear_sonic[training] + B.10b extras"
cd "${REPO_DIR}"
pip install --quiet -e "gear_sonic/[training]"
pip install --quiet "open3d==0.19.0" "tensordict==0.12.1" "vector-quantize-pytorch==1.28.1"
# 2026-09-06: isaaclab_rl's "torch>=2.7" resolves to torch 2.14+cu130 plus the CUDA-13 wheel set
# (nvidia-nccl-cu13, nvidia-cudnn-cu13, cuda-toolkit 13, unsuffixed nvidia-cuda-runtime 13 ...) on
# the ubuntu24.04-cuda12 image (driver 570 / CUDA 12.8): torch.cuda.is_available() False, and even
# after pinning torch the cu13 wheels shadow the cu12 ones (NCCL: "CUDA driver version is
# insufficient for CUDA runtime version"). Drop every non-cu12 NVIDIA wheel, reinstall the cu128
# pair WITH deps (restores nvidia-*-cu12), restore the numpy pin the reinstall bumps.
cu13=$(pip list 2>/dev/null | awk '{print $1}' | grep -E '^(nvidia-.*|cuda-toolkit|cuda-bindings|cuda-pathfinder)$' | grep -v -- '-cu12$' || true)
[[ -n "${cu13}" ]] && pip uninstall -y -q ${cu13}
pip install --quiet --force-reinstall "torch==2.7.0" "torchvision==0.22.0" \
  --index-url https://download.pytorch.org/whl/cu128
pip install --quiet "numpy==1.26.4" "triton==3.3.0"
python -c "import torch, numpy; assert torch.cuda.is_available(), 'torch sees no CUDA after the cu128 pin'; print('torch', torch.__version__, 'nccl', torch.cuda.nccl.version(), 'numpy', numpy.__version__, 'cuda OK')"
step "installed gear_sonic[training] + open3d + tensordict + vector-quantize-pytorch; torch pinned 2.7.0+cu128"

#-------------------------------------------------------------------------------
# Phase 13 — Validation
#-------------------------------------------------------------------------------
log "Phase 13: validation"
step "vulkaninfo (should list one row per GPU under driverName=NVIDIA):"
vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverName" | head -20 \
  || echo "  (vulkaninfo summary empty — Isaac Sim will not see GPUs)"

if [[ -f "${REPO_DIR}/check_environment.py" ]]; then
  step "check_environment.py --training:"
  (cd "${REPO_DIR}" && python check_environment.py --training 2>&1 | tail -20) || true
fi

step "Hydra dry-compose (X2 smoke config):"
(cd "${REPO_DIR}" && python gear_sonic/train_agent_trl.py \
  --config-name=base \
  +exp=manager/universal_token/all_modes/sonic_x2_ultra_smoke \
  --cfg job 2>&1 \
  | grep -E "motion_file|num_envs:|num_learning_iterations|project_name" | head -10) \
  || echo "  (Hydra compose failed — check the gear_sonic[training] install above)"

cat <<EOF

================================================================================
Bootstrap complete.
================================================================================
Next steps:
  1. Build the demo motion bank from the upstream reference clips (or scp
     your own motion-lib corpus to this node and point MOTION_FILE at it):
       cd ${REPO_DIR}
       bash tools/build_demo_bank_from_upstream.sh
  2. Bring your own checkpoints/models if fine-tuning (see MODELS.md); e.g.
       scp -r <workstation>:<run dir>/model_step_NNNNNN.pt ubuntu@<this node>:~/ckpt/
  3. Launch the 8-GPU smoke (200 iterations, W&B off):
       tmux new -d -s smoke "bash gear_sonic/scripts/cloud/run_smoke_8gpu.sh"
       tmux a -t smoke
     or a fine-tune from a checkpoint:
       tmux new -d -s ft "
         NUM_ENVS=8192 NUM_ITERS=4000 EXP_NAME=sonic_x2_ultra \\
         EXTRA_FLAGS='+checkpoint=\$HOME/ckpt/model_step_NNNNNN.pt' \\
         LOG_FILE=\$HOME/ft.log bash gear_sonic/scripts/cloud/run_smoke_8gpu.sh
       "
================================================================================
EOF
