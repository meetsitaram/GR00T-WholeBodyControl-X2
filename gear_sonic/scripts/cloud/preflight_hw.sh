#!/bin/bash
# Hardware pre-flight for a multi-node SONIC launch. Covers the failure
# classes that cost us a smoke each: B1 (fabricmanager down after DKMS
# rebuild -> CUDA 802 -> misleading "doesn't support bf16/gpu") and the
# multinode-doc requirement that NCCL find live IB rails.
#
# Env:
#   ISAACLAB_PYTHON  python of the training env (default: $HOME/miniconda3/envs/env_isaaclab/bin/python)
#   DATA_FILES       space-separated files that must exist (corpora, checkpoints); optional
case "${1:-}" in
  -h|--help) sed -n '2,/^FAIL=/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \{0,1\}//'; exit 0;;
esac
ISAACLAB_PYTHON=${ISAACLAB_PYTHON:-$HOME/miniconda3/envs/env_isaaclab/bin/python}
FAIL=0
say() { printf '  %-34s %s\n' "$1" "$2"; }
bad() { printf '  %-34s \033[31m%s\033[0m\n' "$1" "$2"; FAIL=1; }

echo "===== $(hostname) ====="

DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
[ "$NGPU" = "8" ] && say "GPUs visible" "8 (driver $DRV)" || bad "GPUs visible" "$NGPU (expected 8)"

FM=$(systemctl is-active nvidia-fabricmanager 2>/dev/null)
FMV=$(nv-fabricmanager --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
if [ "$FM" = "active" ]; then
  [ "$FMV" = "$DRV" ] && say "fabricmanager" "active, version matches driver" \
                      || bad "fabricmanager" "active but FM=$FMV != DRV=$DRV"
else
  bad "fabricmanager" "$FM  <-- CUDA 802 incoming (B1)"
fi

# NVLink/NVSwitch present?
NVL=$(nvidia-smi topo -m 2>/dev/null | grep -cE "NV[0-9]+")
[ "$NVL" -gt 0 ] && say "NVLink topology" "$NVL peer links" || bad "NVLink topology" "none found"

# InfiniBand rails (multinode NCCL)
if command -v ibstat >/dev/null 2>&1; then
  ACT=$(ibstat 2>/dev/null | grep -c "State: Active")
  RATE=$(ibstat 2>/dev/null | grep -oE "Rate: [0-9]+" | head -1 | grep -oE "[0-9]+")
  [ "$ACT" -ge 8 ] && say "IB rails active" "$ACT @ ${RATE}Gb/s" || bad "IB rails active" "$ACT (expected 8)"
else
  bad "ibstat" "not installed"
fi

# GPUs idle?
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
[ "$BUSY" = "0" ] && say "GPUs idle" "no compute processes" || bad "GPUs idle" "$BUSY process(es) running"

# The REAL test: actual CUDA allocation on every GPU + bf16 (not just is_available)
"$ISAACLAB_PYTHON" - <<'PY' 2>&1 | tail -4
import torch, sys
try:
    n = torch.cuda.device_count()
    for i in range(n):
        torch.zeros(1024, device=f"cuda:{i}")
    ok = torch.cuda.is_bf16_supported()
    print(f"  {'CUDA alloc all GPUs':<34} {n}/8 OK, bf16={ok}")
    sys.exit(0 if (n == 8 and ok) else 1)
except Exception as e:
    print(f"  {'CUDA alloc all GPUs':<34} FAILED: {type(e).__name__}: {str(e)[:70]}")
    sys.exit(1)
PY
[ ${PIPESTATUS[0]} -ne 0 ] && FAIL=1

FREE=$(df -BG / | tail -1 | awk '{print $4}')
say "disk free" "$FREE"
if [ -n "${DATA_FILES:-}" ]; then
  # shellcheck disable=SC2086  # DATA_FILES is intentionally word-split.
  ls $DATA_FILES >/dev/null 2>&1 && say "data" "all DATA_FILES present" || bad "data" "MISSING one of: $DATA_FILES"
fi

[ $FAIL -eq 0 ] && echo "  ==> PREFLIGHT PASS" || echo "  ==> PREFLIGHT FAIL"
exit $FAIL
